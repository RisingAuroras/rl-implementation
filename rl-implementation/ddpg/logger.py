import logging


"""
这个类初始化时需要传入一个日志名称和一个日志级别，默认是 INFO。
它会创建一个名为 logger 的 logging.Logger 实例，设置日志级别，创建一个格式化器，
以及一个流处理器并将其添加到 logger 实例中。
类中的 info、warning、error、critical 方法是将相应级别的日志消息记录到日志中的便捷方法。使用方法如下：
"""
class Logger:
    def __init__(self, name, level=logging.INFO):


        # 创建一个日志器logger并设置其日志级别为level(默认info)
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        # 创建一个格式器formatter并将其添加到处理器handler
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

        # 创建处理器handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # 为日志器logger添加上面创建的处理器handler
        self.logger.addHandler(console_handler)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

# if __name__ == "__main__":
#     logger = Logger('my_logger', logging.INFO)
#     logger.info('This is an informational message.')
#     logger.warning('This is a warning message.')
#     logger.error('This is an error message.')
#     logger.critical('This is a critical message.')

