"""
Link documents: [https://docs.python.org/3/library/logging.html]

Module logging trong Python dùng để ghi lại các thông tin, sự kiện, cảnh báo, lỗi 
    hoặc các thông báo khác trong quá trình chạy chương trình.

Các mức độ logging phổ biến:
- DEBUG: Thông tin chi tiết phục vụ debug.
- INFO: Thông tin chung về tiến trình chương trình.
- WARNING: Cảnh báo có thể gây lỗi.
- ERROR: Lỗi xảy ra nhưng chương trình vẫn chạy tiếp.
- CRITICAL: Lỗi nghiêm trọng, chương trình có thể dừng.

Ví dụ chi tiết về logging:
"""

# import logging

# # Cấu hình logging cơ bản
# logging.basicConfig(
#     level=logging.DEBUG,  # Ghi tất cả các mức log từ DEBUG trở lên
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )

# # Ghi log với các mức độ khác nhau
# logging.debug("Đây là log DEBUG: dùng để debug chi tiết.")
# logging.info("Đây là log INFO: thông tin chung.")
# logging.warning("Đây là log WARNING: cảnh báo.")
# logging.error("Đây là log ERROR: có lỗi xảy ra.")
# logging.critical("Đây là log CRITICAL: lỗi nghiêm trọng.")


# TRAIN
import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme

custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "critical": "bold white on red",
    }
)
console = Console(theme=custom_theme)

LOGFORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOGFORMAT_RICH = "%(message)s"

logging.basicConfig(
    level=logging.WARNING,  # Chỉ hiện WARNING trở lên
    format=LOGFORMAT_RICH,
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)

# Test log các mức độ
logging.debug("Đây là debug")      # Không hiện
logging.info("Đây là info")        # Không hiện
logging.warning("Đây là cảnh báo")  # Hiện (màu vàng)
logging.error("Đây là lỗi")        # Hiện (màu đỏ)
logging.critical("Lỗi nghiêm trọng")  # Hiện (màu đỏ đậm)
