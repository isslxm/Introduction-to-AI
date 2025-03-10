import numpy as np
import random
import matplotlib.pyplot as plt

# Параметры задачи
num_groups = 4
num_teachers = 4
num_rooms = 4
num_days = 5
max_classes_per_day = 4

# Предметы и их количество занятий в неделю
subjects = {
    "Manas Telling": 1,
    "Advanced Python": 2,
    "English": 4,
    "Data Structures": 4,
    "Introduction to AI": 2,
    "History": 1,
    "Geography": 1
}
subject_list = list(subjects.keys())
num_subjects = len(subject_list)

# Генерация начального расписания
def generate_schedule():
    schedule = np.full((num_days, num_groups, max_classes_per_day), -1, dtype=int)
    
    for group in range(num_groups):
        subject_pool = []
        for subject, count in subjects.items():
            subject_pool.extend([subject_list.index(subject)] * count)
        random.shuffle(subject_pool)
        
        index = 0
        for day in range(num_days):
            daily_subjects = set()
            for class_num in range(max_classes_per_day):
                if index < len(subject_pool):
                    while subject_pool[index] in daily_subjects:
                        random.shuffle(subject_pool)
                    schedule[day, group, class_num] = subject_pool[index]
                    daily_subjects.add(subject_pool[index])
                    index += 1
    
    return schedule

# Функция оценки (fitness)
def fitness(schedule):
    penalty = 0
    
    # Проверка количества занятий
    for group in range(num_groups):
        subject_counts = {subject: 0 for subject in subjects}
        for day in range(num_days):
            for class_num in range(max_classes_per_day):
                subject_index = schedule[day, group, class_num]
                if subject_index != -1:
                    subject_counts[subject_list[subject_index]] += 1
        for subject, required_count in subjects.items():
            penalty += abs(subject_counts[subject] - required_count)
    
    # Проверка на конфликты преподавателей и аудиторий
    for day in range(num_days):
        for class_num in range(max_classes_per_day):
            teachers = set()
            rooms = set()
            for group in range(num_groups):
                subject_index = schedule[day, group, class_num]
                if subject_index != -1:
                    teacher = subject_index % num_teachers
                    room = subject_index % num_rooms
                    if teacher in teachers or room in rooms:
                        penalty += 10
                    teachers.add(teacher)
                    rooms.add(room)
    
    return penalty

# Глобальная переменная для хранения истории fitness
fitness_history = []

def pso(num_particles, max_iter):
    global fitness_history
    fitness_history = []  # Reset history at the start
    
    # Initialize particles
    particles = [generate_schedule() for _ in range(num_particles)]
    velocities = [np.zeros_like(particles[0]) for _ in range(num_particles)]
    personal_best = [p.copy() for p in particles]
    personal_best_fitness = [fitness(p) for p in personal_best]
    
    # Find global best
    best_idx = np.argmin(personal_best_fitness)
    global_best = personal_best[best_idx].copy()
    global_best_fitness = personal_best_fitness[best_idx]
    
    # PSO parameters
    w = 0.5  # Inertia weight
    c1 = 1.5  # Cognitive coefficient
    c2 = 1.5  # Social coefficient
    
    for iteration in range(max_iter):
        for i in range(num_particles):
            # Update velocity
            r1, r2 = random.random(), random.random()
            
            # Element-wise operations on the arrays
            cognitive = c1 * r1 * (personal_best[i] - particles[i])
            social = c2 * r2 * (global_best - particles[i])
            
            velocities[i] = w * velocities[i] + cognitive + social
            
            # Update particle positions
            new_particle = particles[i] + np.round(velocities[i]).astype(int)
            
            # Ensure new particle has valid subject indices
            new_particle = np.clip(new_particle, -1, num_subjects - 1).astype(int)
            particles[i] = new_particle
            
            # Update personal best
            current_fitness = fitness(particles[i])
            if current_fitness < personal_best_fitness[i]:
                personal_best[i] = particles[i].copy()
                personal_best_fitness[i] = current_fitness
                
                # Update global best if needed
                if current_fitness < global_best_fitness:
                    global_best = particles[i].copy()
                    global_best_fitness = current_fitness
        
        # Record fitness history
        fitness_history.append(global_best_fitness)
        print(f"Iteration {iteration}, Best Fitness: {global_best_fitness}")
    
    return global_best

# Функция для вывода расписания
def print_schedule(schedule):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    print("\nBest Schedule:")
    for day in range(num_days):
        print(f"\n{days[day]}:")
        print("Group\tClass 1\t\tClass 2\t\tClass 3\t\tClass 4")
        print("-" * 60)
        for group in range(num_groups):
            print(f"Group {group + 1}:", end="\t")
            for class_num in range(max_classes_per_day):
                subject_index = schedule[day, group, class_num]
                if subject_index != -1:
                    subject_name = subject_list[subject_index]
                    print(f"{subject_name:<15}", end="\t")
                else:
                    print("None\t\t", end="")
            print()
        print("-" * 60)

# Запуск PSO с меньшим количеством итераций для быстрой проверки
best_schedule = pso(num_particles=50, max_iter=100)
print_schedule(best_schedule)

# Построение графика
plt.figure(figsize=(10, 5))
plt.plot(range(len(fitness_history)), fitness_history, marker='o', linestyle='-', color='b', label="Fitness Over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Fitness Score")
plt.title("Fitness Improvement Over Iterations")
plt.legend()
plt.grid()
plt.show()