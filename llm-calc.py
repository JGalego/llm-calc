r"""
  _      _      __  __    _____      _            _       _
 | |    | |    |  \/  |  / ____|    | |          | |     | |
 | |    | |    | \  / | | |     __ _| | ___ _   _| | __ _| |_ ___  _ __
 | |    | |    | |\/| | | |    / _` | |/ __| | | | |/ _` | __/ _ \| '__|
 | |____| |____| |  | | | |___| (_| | | (__| |_| | | (_| | || (_) | |
 |______|______|_|  |_|  \_____\__,_|_|\___|\__,_|_|\__,_|\__\___/|_|

Learn how to size LLM workloads - find out how much compute power and storage you need to train your model!
"""

from st_pages import Page, show_pages, add_page_title

# Adds the title and icon to the current page
add_page_title()

# Specify what pages should be shown in the sidebar,
# and what their titles and icons should be
show_pages(
    [
        Page("pages/training.py", "Pre-Training", ":baby:"),
        Page("pages/inference.py", "Inference", ":rocket:"),
    ]
)
