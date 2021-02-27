import logging


# Heavily adapted from: https://stackoverflow.com/a/31695996/866930
def setup_logger(log_name, log_file, level=logging.INFO, attach_stream_handler=False):
    log = logging.getLogger(log_name)

    # Configure the format of the messages that will be written in
    # the log. We use a verbose format since that is more informative
    # for debugging purposes.
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s', 
        datefmt='%m/%d/%Y %I:%M:%S %p'
    )

    # Configure how the log file should be written.
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    if attach_stream_handler:
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)

    # Finally, add the handlers to the log and it's ready for use.
    log.setLevel(level)
    log.addHandler(fileHandler)
    if attach_stream_handler:
        log.addHandler(streamHandler)
