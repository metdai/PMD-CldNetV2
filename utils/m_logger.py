# %%
import logging

"""
Created on Wed June 14 16:18:10 2022
封装logger记录器, 方便使用直接调用
@author: BEOH
@email: beoh86@yeah.net
"""


def m_logger(log_file, logger_config):
    """
    输入:
        - log_file: log记录文件
        - logger_config: log配置\n
    输出:
        - logger对象
    """
    # 1产生日志对象
    logger = logging.getLogger("logger")
    if not logger.handlers:
        # 2定义handler对象
        terminal_handler = logging.StreamHandler()
        # terminal_handler.setLevel(20)           # handler设置日志级别
        file_handler = logging.FileHandler(
            log_file, mode=logger_config["mode"], encoding='UTF-8'
        )
        # 3定义formatter对象
        file_formatter = logging.Formatter(
            fmt=logger_config["file_formatter"]["fmt"],
            datefmt=logger_config["file_formatter"]["datefmt"],
        )
        terminal_formatter = logging.Formatter(
            fmt=logger_config["terminal_formatter"]["fmt"],
            datefmt=logger_config["terminal_formatter"]["datefmt"],
        )
        # 4handler对象设置输出模版
        file_handler.setFormatter(file_formatter)
        terminal_handler.setFormatter(terminal_formatter)
        # 5logger对象添加handler日志处理器
        logger.addHandler(file_handler)
        logger.addHandler(terminal_handler)
    # 6logger对象设置日志级别
    logger.setLevel(logger_config["setLevel"])
    return logger



