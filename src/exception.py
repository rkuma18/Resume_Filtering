import sys
import logging
import traceback

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = self._get_detailed_error_message(error_message, error_detail)
        logging.error(self.error_message)  # Automatically log error when raised

    def _get_detailed_error_message(self, error_message, error_detail: sys) -> str:
        try:
            exc_type, exc_obj, exc_tb = error_detail.exc_info()
            file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "Unknown"
            line_number = exc_tb.tb_lineno if exc_tb else "Unknown"
            return f"❌ Error in [{file_name}] at line [{line_number}]: {error_message}"
        except Exception as e:
            return f"❌ CustomException failed to extract traceback: {error_message} | {str(e)}"

    def __str__(self):
        return self.error_message
