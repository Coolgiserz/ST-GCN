# !/usr/bin/env python
# -*- coding:utf-8 -*-
# coding=utf-8

import logging
import os


class Logger(object):
    def __init__(self, logPath, use_console=False):
        """
        initial
        """
        logging.addLevelName(20, "NOTICE:")
        logging.addLevelName(30, "WARNING:")
        logging.addLevelName(40, "FATAL:")
        logging.addLevelName(50, "FATAL:")
        logging.basicConfig(level=logging.DEBUG,
                            format="%(levelname)s %(filename)s %(asctime)s\t%(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S",
                            filename=logPath,
                            filemode="w")

        if use_console:
            console = logging.StreamHandler()
            console.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(levelname)s [pid:%(process)s] %(message)s")
            console.setFormatter(formatter)
            logging.getLogger("").addHandler(console)

    def debug(self, msg=""):
        """
        output DEBUG level LOG
        """
        logging.debug(str(msg))

    def info(self, msg=""):
        """
        output INFO level LOG
        """
        logging.info(str(msg))

    def warning(self, msg=""):
        """
        output WARN level LOG
        """
        logging.warning(str(msg))

    def exception(self, msg=""):
        """
        output Exception stack LOG
        """
        logging.exception(str(msg))

    def error(self, msg=""):
        """
        output ERROR level LOG
        """
        logging.error(str(msg))

    def critical(self, msg=""):
        """
        output FATAL level LOG
        """
        logging.critical(str(msg))


if __name__ == "__main__":
  testlog = Logger("oupput.log")
  testlog.info("info....")
  testlog.warning("warning....")
  testlog.critical("critical....")
  try:
    lists = []
    print(lists[1])
  except Exception as ex:
    """logging.exception()输出格式： 
    FATAL: [pid:7776] execute task failed. the exception as follows: 
    Traceback (most recent call last): 
      File "logtool.py", line 86, in <module> 
        print lists[1] 
    IndexError: list index out of range 
    """
    testlog.exception("execute task failed. the exception as follows:")
    testlog.info("++++++++++++++++++++++++++++++++++++++++++++++")
    """logging.error()输出格式： 
    FATAL: [pid:7776] execute task failed. the exception as follows: 
    """
    testlog.error("execute task failed. the exception as follows:")
    exit(1)