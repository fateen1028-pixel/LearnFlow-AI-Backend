from bson import ObjectId
from app.models.plan import Plan
from app.models.todo import Todo
from app.utils.helpers import get_db
from datetime import datetime
from app.utils.ai_helpers import run_chain, roadmap_prompt, refinement_prompt_template
import json

class PlanService:
    @staticmethod
    def generate_roadmap(data):
        required_fields = ["topic", "days", "hours", "experience"]
        if not data or any(not data.get(field) for field in required_fields):
            return {
                "status": "error",
                "message": "Missing required fields (topic, days, hours, or experience). Please complete the form."
            }, 400
        
        roadmap_data = run_chain(roadmap_prompt, data)
        
        if roadmap_data:
            return {"status": "success", "roadmap": roadmap_data}
        else:
            return {"status": "error", "message": "Failed to generate roadmap (AI response invalid)"}, 500

    @staticmethod
    def generate_todo_list(user_id, roadmap):
        if not roadmap:
            return {"status": "error", "message": "Missing roadmap"}, 400

        try:
            plans_col = get_db().learning_plans
            todos_col = get_db().todos
            
            plan_doc = Plan.create_plan_doc(
                user_id, 
                roadmap.get("topic"), 
                roadmap.get("days"), 
                roadmap.get("hours")
            )
            plan_result = plans_col.insert_one(plan_doc)
            plan_id = plan_result.inserted_id

            for day in roadmap.get("roadmap", []):
                day_num = day.get("day")
                for task in day.get("tasks", []):
                    parent_task_title = task.get("parent_task")
                    for sub_task in task.get("sub_tasks", []):
                        todo_doc = Todo.create_todo_doc(
                            user_id, 
                            str(plan_id), 
                            day_num, 
                            parent_task_title,
                            sub_task.get("task"),
                            sub_task.get("duration_minutes"),
                            sub_task.get("description")
                        )
                        todos_col.insert_one(todo_doc)

            return {
                "status": "success",
                "message": "Todo list generated successfully",
                "planId": str(plan_id)
            }

        except Exception as e:
            print(f"Error generating todo: {e}")
            return {"status": "error", "message": "Failed to generate todo list"}, 500

    @staticmethod
    def refine_roadmap(roadmap, instruction):
        prompt_data = {
            "roadmap": json.dumps(roadmap),
            "instruction": instruction
        }

        refined_roadmap = run_chain(refinement_prompt_template, prompt_data)

        if refined_roadmap:
            return {"status": "success", "roadmap": refined_roadmap}
        else:
            return {"status": "error", "message": "Failed to refine roadmap"}, 500

    @staticmethod
    def get_active_plans(user_id):
        try:
            plans_col = get_db().learning_plans
            
            active_plans = list(plans_col.find(
                {"userId": user_id, "status": {"$ne": "COMPLETED"}},
                {"_id": 1, "topic": 1, "days": 1, "hours": 1, "progress": 1, "status": 1, "startDate": 1}
            ).sort("_id", -1))
            
            for plan in active_plans:
                plan['_id'] = str(plan['_id'])
            
            return {
                "status": "success",
                "plans": active_plans
            }
        except Exception as e:
            print(f"Error fetching active plans: {e}")
            return {"status": "error", "message": "Failed to fetch plans"}, 500

    @staticmethod
    def get_all_plans(user_id):
        try:
            plans_col = get_db().learning_plans
            
            all_plans = list(plans_col.find(
                {"userId": user_id},
                {"_id": 1, "topic": 1, "days": 1, "hours": 1, "progress": 1, "status": 1, "startDate": 1}
            ).sort("_id", -1))
            
            for plan in all_plans:
                plan['_id'] = str(plan['_id'])
            
            return {
                "status": "success",
                "plans": all_plans
            }
        except Exception as e:
            print(f"Error fetching all plans: {e}")
            return {"status": "error", "message": "Failed to fetch plans"}, 500

    @staticmethod
    def delete_plan(user_id, plan_id):
        try:
            plans_col = get_db().learning_plans
            todos_col = get_db().todos
            
            # Verify plan belongs to user before deleting
            plan = plans_col.find_one({"_id": ObjectId(plan_id), "userId": user_id})
            if not plan:
                return {"status": "error", "message": "Plan not found"}, 404
                
            result = plans_col.delete_one({"_id": ObjectId(plan_id), "userId": user_id})
            
            if result.deleted_count == 0:
                return {"status": "error", "message": "Plan not found"}, 404
            
            todos_col.delete_many({"planId": plan_id, "userId": user_id})
            
            return {
                "status": "success",
                "message": "Plan and all associated todos deleted successfully"
            }
        except Exception as e:
            print(f"Error deleting plan: {e}")
            return {"status": "error", "message": "Failed to delete plan"}, 500

    @staticmethod
    def check_initial_data(user_id):
        try:
            plans_col = get_db().learning_plans
            
            # Get latest plan for this user
            latest_plan = plans_col.find_one(
                {"userId": user_id}, 
                sort=[('_id', -1)]
            )
            plan_id = str(latest_plan['_id']) if latest_plan else None

            # Get active plans for this user
            active_plans = list(plans_col.find(
                {"userId": user_id, "status": {"$ne": "COMPLETED"}},
                {"_id": 1, "topic": 1, "days": 1, "hours": 1, "progress": 1, "status": 1, "startDate": 1}
            ).sort("_id", -1))
            
            for plan in active_plans:
                plan['_id'] = str(plan['_id'])

            return {
                "status": "success",
                "planId": plan_id,
                "activePlans": active_plans
            }
        except Exception as e:
            print(f"Database error during initial data check: {e}")
            return {"status": "error", "message": "Failed to connect to database"}, 500
        

    # In plan_service.py - modify get_next_day_task
    @staticmethod
    def get_next_day_task(user_id, plan_id):
        try:
            plans_col = get_db().learning_plans
            todos_col = get_db().todos
            
            # Get current plan
            plan = plans_col.find_one({"_id": ObjectId(plan_id), "userId": user_id})
            if not plan:
                return {"status": "error", "message": "Plan not found"}, 404
            
            current_day = plan.get('currentDay', 1)
            total_days = plan.get('days', 1)
            
            # Check if next day exists
            next_day = current_day + 1
            if next_day > total_days:
                return {
                    "status": "success", 
                    "task": None, 
                    "message": "No more days available in this plan"
                }
            
            # Find the first task from the next day
            next_task = todos_col.find_one({
                "planId": plan_id,
                "userId": user_id,
                "day": next_day
            }, sort=[("createdAt", 1)])  # Get the first task by creation order
            
            if not next_task:
                return {
                    "status": "success",
                    "task": None,
                    "message": "No tasks available for next day"
                }
            
            # Create a copy of the task for the current day (not modifying original)
            import copy
            bonus_task = copy.deepcopy(next_task)
            
            # Insert as a new task for current day
            bonus_task['_id'] = ObjectId()  # New ID for the bonus task
            bonus_task['day'] = current_day
            bonus_task['originalDay'] = next_day
            bonus_task['isBonus'] = True
            bonus_task['bonusFromDay'] = next_day
            bonus_task['createdAt'] = datetime.now()
            bonus_task['updatedAt'] = datetime.now()
            
            # Insert the bonus task
            todos_col.insert_one(bonus_task)
            
            # Convert for response
            bonus_task['_id'] = str(bonus_task['_id'])
            
            return {
                "status": "success",
                "task": bonus_task,
                "message": f"Added first task from day {next_day} as bonus",
                "bonusFromDay": next_day,
                "currentDay": current_day  # Stay on same day
            }
            
        except Exception as e:
            print(f"Error in get_next_day_task: {e}")
            return {"status": "error", "message": f"Failed to fetch next day task: {str(e)}"}, 500