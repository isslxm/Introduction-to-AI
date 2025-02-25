import numpy as np
import random
import matplotlib.pyplot as plt

# Параметры задачи
num_groups = 4
num_teachers = 4
num_rooms = 4
num_days = 5
max_classes_per_day = 3

# Предметы и их количество занятий в неделю
subjects = {
    "Manas Telling": 1,
    "Advanced Python": 2,
    "English": 5,
    "Data Structures": 3,
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
                        penalty += 1
                    teachers.add(teacher)
                    rooms.add(room)
    
    return penalty

# PSO алгоритм
def pso(num_particles, max_iter):
    particles = [generate_schedule() for _ in range(num_particles)]
    velocities = [np.zeros_like(particles[0]) for _ in range(num_particles)]
    personal_best = [p.copy() for p in particles]
    global_best = min(personal_best, key=fitness)
    
    for iteration in range(max_iter):
        for i in range(num_particles):
            velocities[i] = velocities[i] * 0.5 + random.random() * (personal_best[i] - particles[i]) + random.random() * (global_best - particles[i])
            particles[i] = np.clip(particles[i] + velocities[i], 0, num_subjects - 1).astype(int)
            
            if fitness(particles[i]) < fitness(personal_best[i]):
                personal_best[i] = particles[i].copy()
            if fitness(personal_best[i]) < fitness(global_best):
                global_best = personal_best[i].copy()
        print(f"Iteration {iteration}, Best Fitness: {fitness(global_best)}")
    
    return global_best

# Вывод расписания
def print_schedule(schedule):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    print("\nBest Schedule:")
    for day in range(num_days):
        print(f"\n{days[day]}:")
        print("Group\tClass 1\t  Class 2\t  Class 3")
        print("-" * 40)
        for group in range(num_groups):
            print(f"Group {group + 1}:", end="\t")
            for class_num in range(max_classes_per_day):
                subject_index = schedule[day, group, class_num]
                subject_name = subject_list[subject_index]
                print(f"{subject_name}", end="\t")
            print()
        print("-" * 40)

# Запуск PSO
best_schedule = pso(num_particles=30, max_iter=100)
print_schedule(best_schedule)


# Глобальная переменная для хранения истории fitness
fitness_history = []

def pso(num_particles, max_iter):
    global fitness_history
    particles = [generate_schedule() for _ in range(num_particles)]
    velocities = [np.zeros_like(particles[0]) for _ in range(num_particles)]
    personal_best = [p.copy() for p in particles]
    global_best = min(personal_best, key=fitness)
    
    for iteration in range(max_iter):
        for i in range(num_particles):
            velocities[i] = velocities[i] * 0.5 + random.random() * (personal_best[i] - particles[i]) + random.random() * (global_best - particles[i])
            particles[i] = np.clip(particles[i] + velocities[i], 0, num_subjects - 1).astype(int)
            
            if fitness(particles[i]) < fitness(personal_best[i]):
                personal_best[i] = particles[i].copy()
            if fitness(personal_best[i]) < fitness(global_best):
                global_best = personal_best[i].copy()
        
        # Сохраняем текущее значение fitness для графика
        best_fitness = fitness(global_best)
        fitness_history.append(best_fitness)
        print(f"Iteration {iteration}, Best Fitness: {best_fitness}")
    
    return global_best

# Запуск PSO
best_schedule = pso(num_particles=30, max_iter=100)

# Построение графика
plt.figure(figsize=(10, 5))
plt.plot(range(len(fitness_history)), fitness_history, marker='o', linestyle='-', color='b', label="Fitness Over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Fitness Score")
plt.title("Fitness Improvement Over Iterations")
plt.legend()
plt.grid()
plt.show()
