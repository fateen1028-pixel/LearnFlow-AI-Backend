from datetime import datetime, timedelta
from bson import ObjectId
from app.utils.helpers import get_db

class DashboardService:
    @staticmethod
    def get_dashboard_data(user_id, time_range='week'):
        try:
            plans_col = get_db().learning_plans
            todos_col = get_db().todos
            
            # Get all plans for the user
            all_plans = list(plans_col.find({"userId": user_id}))
            
            plans_progress = []
            total_completed_todos = 0
            total_todos = 0
            completed_plans = 0
            active_plans = 0
            
            # Calculate progress for each plan
            for plan in all_plans:
                plan_id = str(plan['_id'])
                
                # Get todos for this plan
                todos = list(todos_col.find({"planId": plan_id, "userId": user_id}))
                total_plan_todos = len(todos)
                completed_plan_todos = len([t for t in todos if t.get('completed', False)])
                
                # Calculate completion rate
                completion_rate = 0
                if total_plan_todos > 0:
                    completion_rate = round((completed_plan_todos / total_plan_todos) * 100)
                
                # Calculate completed days
                completed_days = len(set([t['day'] for t in todos if t.get('completed', False)]))
                
                total_completed_todos += completed_plan_todos
                total_todos += total_plan_todos
                
                if plan.get('status') == 'COMPLETED':
                    completed_plans += 1
                else:
                    active_plans += 1
                
                plans_progress.append({
                    "planId": plan_id,
                    "topic": plan.get('topic', 'Unknown Topic'),
                    "totalDays": plan.get('days', 0),
                    "completedDays": completed_days,
                    "totalTodos": total_plan_todos,
                    "completedTodos": completed_plan_todos,
                    "completionRate": completion_rate,
                    "status": plan.get('status', 'ONGOING'),
                    "startDate": plan.get('startDate'),
                    "dailyHours": plan.get('hours', 1),  # Add daily hours for efficiency calculation
                    "lastActive": plan.get('updatedAt', datetime.now()).strftime("%Y-%m-%d") if plan.get('updatedAt') else datetime.now().strftime("%Y-%m-%d")
                })
            
            # Calculate overall statistics
            overall_completion_rate = 0
            if total_todos > 0:
                overall_completion_rate = round((total_completed_todos / total_todos) * 100)
            
            # Calculate consistency (active days vs total possible days)
            total_active_days = sum([p['completedDays'] for p in plans_progress])
            total_possible_days = sum([p['totalDays'] for p in plans_progress])
            
            consistency = 0
            if total_possible_days > 0:
                consistency = round((total_active_days / total_possible_days) * 100)
            
            # Calculate total study hours (based on completed days * daily hours)
            total_study_hours = sum([p.get('dailyHours', 1) * p['completedDays'] for p in plans_progress])
            
            # Get user's first plan date for "learning since"
            first_plan = plans_col.find_one({"userId": user_id}, sort=[("createdAt", 1)])
            learning_since = "Recently"
            if first_plan and first_plan.get('createdAt'):
                learning_since = first_plan['createdAt'].strftime("%Y-%m-%d")
            
            # Calculate current streak (consecutive days with at least one completed todo)
            current_streak = 1  # Default
            
            # Calculate average daily progress
            avg_daily_progress = 0
            if total_active_days > 0:
                avg_daily_progress = round(overall_completion_rate / total_active_days)
            
            # Calculate productivity score (weighted combination of metrics)
            productivity_score = round((overall_completion_rate * 0.4) + (consistency * 0.3) + (avg_daily_progress * 0.3))
            
            # Generate recent activity
            recent_activity = []
            for plan in all_plans[:3]:
                recent_todos = list(todos_col.find(
                    {"planId": str(plan['_id']), "userId": user_id, "completed": True}
                ).sort("updatedAt", -1).limit(2))
                
                for todo in recent_todos:
                    # Calculate time ago
                    time_diff = datetime.now() - todo.get('updatedAt', datetime.now())
                    if time_diff.days == 0:
                        if time_diff.seconds < 3600:
                            timestamp = f"{time_diff.seconds // 60} minutes ago"
                        else:
                            timestamp = f"{time_diff.seconds // 3600} hours ago"
                    elif time_diff.days == 1:
                        timestamp = "Yesterday"
                    else:
                        timestamp = f"{time_diff.days} days ago"
                    
                    recent_activity.append({
                        "id": str(todo['_id']),
                        "type": "completed",
                        "description": f"Completed: {todo.get('task', 'Task')}",
                        "timestamp": timestamp,
                        "plan": plan.get('topic', 'Unknown Topic')
                    })
            
            # Add plan creation activities
            for plan in all_plans[:2]:
                time_diff = datetime.now() - plan.get('createdAt', datetime.now())
                timestamp = "Recently"
                if time_diff.days > 0:
                    timestamp = f"{time_diff.days} days ago"
                
                recent_activity.append({
                    "id": str(plan['_id']),
                    "type": "created",
                    "description": f"Started: {plan.get('topic', 'New Plan')}",
                    "timestamp": timestamp,
                    "plan": plan.get('topic', 'Unknown Topic')
                })
            
            # Sort activities by timestamp (newest first)
            recent_activity.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # Generate weekly activity data
            weekly_stats = {
                "monday": 0,
                "tuesday": 0,
                "wednesday": 0,
                "thursday": 0,
                "friday": 0,
                "saturday": 0,
                "sunday": 0
            }
            
            # Get completed todos for the current week
            today = datetime.now()
            start_of_week = today - timedelta(days=today.weekday())
            
            for day_offset in range(7):
                day_date = start_of_week + timedelta(days=day_offset)
                day_name = day_date.strftime("%A").lower()
                
                # Count completed todos for this day
                completed_today = todos_col.count_documents({
                    "userId": user_id,
                    "completed": True,
                    "updatedAt": {
                        "$gte": day_date.replace(hour=0, minute=0, second=0),
                        "$lt": day_date.replace(hour=23, minute=59, second=59)
                    }
                })
                
                # Convert todo count to hours (assuming 1 hour per todo)
                weekly_stats[day_name] = completed_today
            
            # Calculate learning efficiency
            learning_efficiency = 0
            if total_study_hours > 0:
                learning_efficiency = min(100, round((overall_completion_rate / total_study_hours) * 10))
            
            # Calculate weekly goals
            weekly_goals = {
                "completed": sum(weekly_stats.values()),
                "total": 10,  # Default goal: 10 todos per week
                "percentage": min(100, round((sum(weekly_stats.values()) / 10) * 100))
            }
            
            # Calculate leaderboard rank (simulated - would be real in production)
            leaderboard_rank = {
                "rank": max(1, 100 - ((overall_completion_rate + consistency + learning_efficiency) // 3)),
                "change": 3  # Simulated positive change
            }
            
            # Calculate achievements
            achievements = []
            
            # First Steps: User has at least one plan
            achievements.append({
                "id": 1,
                "name": "First Steps",
                "icon": "👣",
                "earned": len(all_plans) > 0
            })
            
            # Early Bird: User has completed todos early in the day
            early_bird = False
            early_todos = list(todos_col.find({
                "userId": user_id,
                "completed": True,
                "updatedAt": {
                    "$gte": datetime.now().replace(hour=6, minute=0, second=0),
                    "$lt": datetime.now().replace(hour=9, minute=0, second=0)
                }
            }).limit(1))
            if early_todos:
                early_bird = True
            
            achievements.append({
                "id": 2,
                "name": "Early Bird",
                "icon": "🐦",
                "earned": early_bird
            })
            
            # Consistent Learner: User has consistency > 50%
            achievements.append({
                "id": 3,
                "name": "Consistent Learner",
                "icon": "📚",
                "earned": consistency > 50
            })
            
            # Task Master: User has completed at least 10 todos
            achievements.append({
                "id": 4,
                "name": "Task Master",
                "icon": "✅",
                "earned": total_completed_todos >= 10
            })
            
            # Week Warrior: User has completed todos on 5+ days in a week
            week_warrior = False
            if total_active_days >= 5:
                week_warrior = True
            
            achievements.append({
                "id": 5,
                "name": "Week Warrior",
                "icon": "⚔️",
                "earned": week_warrior
            })
            
            # Create dashboard data
            dashboard_data = {
                "plansProgress": plans_progress,
                "overallStats": {
                    "completionRate": overall_completion_rate,
                    "consistency": consistency,
                    "totalCompletedTodos": total_completed_todos,
                    "activeDays": total_active_days,
                    "totalPlans": len(all_plans),
                    "activePlans": active_plans,
                    "completedPlans": completed_plans,
                    "totalStudyHours": total_study_hours,
                    "avgDailyProgress": avg_daily_progress,
                    "currentStreak": current_streak,
                    "learningSince": learning_since,
                    "productivityScore": productivity_score,
                    "learningEfficiency": learning_efficiency,
                    "weeklyGoals": weekly_goals,
                    "leaderboardRank": leaderboard_rank
                },
                "recentActivity": recent_activity,
                "weeklyStats": weekly_stats,
                "achievements": achievements
            }
            
            return {
                "status": "success",
                "dashboard": dashboard_data,
                "timeRange": time_range
            }
            
        except Exception as e:
            print(f"Error fetching dashboard data: {e}")
            return {
                "status": "error",
                "message": "Failed to load dashboard data"
            }, 500