"""Task executor: dispatch --task 2,3,4,5 to experiments.taskN.run(args)."""


class TaskExecutor:
    def run(self, task_id, args):
        if task_id == 2:
            from experiments import task2
            return task2.run(args)
        if task_id == 3:
            from experiments import task3
            return task3.run(args)
        if task_id == 4:
            from experiments import task4
            return task4.run(args)
        if task_id == 5:
            from experiments import task5
            return task5.run(args)
        raise ValueError(f"Unknown task_id: {task_id}. Use 2, 3, 4, or 5.")
