import numpy as np
import random

# Problem parameters
courses = [
    'Advanced Python',   # 0
    'Manas Telling',      # 1
    'Kyrgyz Language',    # 2
    'History',            # 3
    'Geography',          # 4
    'AI Introduction',    # 5
    'English'             # 6
]

course_teacher = [1, 0, 0, 2, 3, 4, 5]  # Teacher index for each course
teachers = ['T0', 'T1', 'T2', 'T3', 'T4', 'T5']
groups = [21, 22, 23, 24]
num_groups = 4
num_days = 5  # Updated for a full week
num_periods_per_day = 4
num_periods_total = num_days * num_periods_per_day
num_rooms = 4
num_courses = len(courses)

# PSO parameters
num_particles = 100
max_iter = 500
w = 0.7
c1 = 1.4
c2 = 1.4

class Particle:
    def __init__(self):
        self.position = np.random.rand(num_groups * num_periods_total * 2) * 10 - 5
        self.velocity = np.zeros(num_groups * num_periods_total * 2)
        self.best_position = self.position.copy()
        self.best_score = float('inf')

    def update_velocity(self, global_best):
        self.velocity = (w * self.velocity +
                         c1 * random.random() * (self.best_position - self.position) +
                         c2 * random.random() * (global_best - self.position))

    def update_position(self):
        self.position += self.velocity


def decode_position(position):
    schedule = {}
    for group in range(num_groups):
        group_schedule = []
        for day in range(num_days):
            day_schedule = []
            for period in range(num_periods_per_day):
                idx = (group * num_periods_total + (day * num_periods_per_day + period)) * 2
                course = int(round(position[idx])) % num_courses
                room = int(round(position[idx + 1])) % num_rooms
                teacher = course_teacher[course]
                day_schedule.append((course, teacher, room))
            group_schedule.append(day_schedule)
        schedule[groups[group]] = group_schedule
    return schedule


def calculate_penalty(schedule):
    penalty = 0
    
    for group, week_schedule in schedule.items():
        for day_schedule in week_schedule:
            courses = [p[0] for p in day_schedule]
            unique_courses = len(set(courses))
            penalty += (len(courses) - unique_courses) * 10
    
    for day in range(num_days):
        for period in range(num_periods_per_day):
            teachers = []
            rooms = []
            for group in groups:
                course, teacher, room = schedule[group][day][period]
                teachers.append(teacher)
                rooms.append(room)
            
            teacher_counts = {t: teachers.count(t) for t in set(teachers)}
            room_counts = {r: rooms.count(r) for r in set(rooms)}
            
            penalty += sum((count - 1) * 5 for count in teacher_counts.values() if count > 1)
            penalty += sum((count - 1) * 5 for count in room_counts.values() if count > 1)
    
    return penalty


def pso():
    particles = [Particle() for _ in range(num_particles)]
    global_best = particles[0].position.copy()
    global_best_score = float('inf')

    for _ in range(max_iter):
        for particle in particles:
            decoded = decode_position(particle.position)
            current_score = calculate_penalty(decoded)

            if current_score < particle.best_score:
                particle.best_score = current_score
                particle.best_position = particle.position.copy()

            if current_score < global_best_score:
                global_best_score = current_score
                global_best = particle.position.copy()

            particle.update_velocity(global_best)
            particle.update_position()

        if global_best_score == 0:
            break

    return decode_position(global_best), global_best_score


def print_schedule(schedule):
    for group in groups:
        print(f"\nGroup {group} Schedule:")
        print(f"{'Day':<10}{'Period':<10}{'Course':<20}{'Teacher':<10}{'Room':<5}")
        for day in range(num_days):
            print(f"Day {day + 1}:")
            for period in range(num_periods_per_day):
                course_idx, teacher_idx, room = schedule[group][day][period]
                course = courses[course_idx]
                teacher = teachers[teacher_idx]
                print(f"{'':<10}{period + 1:<10}{course:<20}{teacher:<10}{room + 1:<5}")

# Run PSO and display results
best_schedule, score = pso()
if score == 0:
    print("Valid schedule found:")
    print_schedule(best_schedule)
else:
    print(f"No perfect schedule found (best score: {score}). Showing best attempt:")
    print_schedule(best_schedule)
