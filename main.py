from utils.args import get_args
from trainer import Trainer
if __name__ == '__main__':
    args = get_args()
    trainer = Trainer(args)

    for task_id in range(len(args.task_list)):
        trainer.incremental_train(task_id)
    for task_id in range(len(args.task_list)):
        trainer.incremental_test(task_id)