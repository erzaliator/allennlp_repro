import logging
import os
import wandb


class Logger():
    def __init__(self, MODEL_DIR, wandb, wandb_flag=True):
        self.wandb_flag = wandb_flag
        if not wandb_flag:
            logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
            rootLogger = logging.getLogger()

            logPath = MODEL_DIR
            isExist = os.path.exists(logPath)
            if not isExist:
                os.makedirs(logPath)
                rootLogger.debug("Logging and wandb init to ", logPath)
            # else:
            #     logger.debug("FATAL: folder already exists")
            #     exit(0)
            fileName = 'out'

            fileHandler = logging.FileHandler("{0}/{1}.log".format(logPath, fileName))
            fileHandler.setFormatter(logFormatter)
            rootLogger.addHandler(fileHandler)

            consoleHandler = logging.StreamHandler()
            consoleHandler.setFormatter(logFormatter)
            rootLogger.addHandler(consoleHandler)

            self.logger =  rootLogger
            self.path = MODEL_DIR
        else:
            self.logger =  wandb

    def debug(self, message):
        if not self.wandb_flag:
            print(message)
        else:
            print(message)
            # self.logger.log({'MESSAGE': message})