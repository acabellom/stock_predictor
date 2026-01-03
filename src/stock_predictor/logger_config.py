import logging

class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[94m",    
        "INFO": "\033[92m",     
        "WARNING": "\033[93m",  
        "ERROR": "\033[91m",    
        "CRITICAL": "\033[41m", 
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"

handler = logging.StreamHandler()
formatter = ColorFormatter("%(asctime)s | %(levelname)s | %(message)s")
handler.setFormatter(formatter)

logger = logging.getLogger("my_app")
logger.addHandler(handler)
logger.setLevel(logging.INFO)

