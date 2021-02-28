import logging


# Heavily adapted from: https://stackoverflow.com/a/31695996/866930
def setup_logger(log_name, log_filename, level=logging.INFO, attach_stream_handler=False):
    """This is an important function as it provides the ability for the user to 
    create and use custom logs. This particular implementation allows one to 
    separate logging to a file and a stream (usually stdout).
    
    @param log_name (str):                  The unique name of the log. Accessible
                                            globally once created.

    @param log_filename (str):              The name / path of the log file that 
                                            will be written.

    @param level (enum):                    The logging level to support. Default 
                                            (INFO).

    @param attach_stream_handler (bool):    Whether or not to write to a stream
                                            (usually stdout).

    @return None
    """
    log = logging.getLogger(log_name)

    # Configure the format of the messages that will be written in
    # the log. We use a verbose format since that is more informative
    # for debugging purposes.
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s', 
        datefmt='%m/%d/%Y %I:%M:%S %p'
    )

    # Configure how the log file should be written.
    fileHandler = logging.FileHandler(log_filename, mode='w')
    fileHandler.setFormatter(formatter)

    # Whether or not to output to a stream (usually stdout).
    if attach_stream_handler:
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)

    # Finally, add the handlers to the log and it's ready for use.
    log.setLevel(level)
    log.addHandler(fileHandler)
    if attach_stream_handler:
        log.addHandler(streamHandler)
