"""phase2/ のモジュールを tests/ から import できるようにする。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
