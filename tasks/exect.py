import os, sys
from pathlib import Path
# 创建的目录
path = "E:/RID/videoexec/person-reid-lib/tasks/output/log"
path=Path(path)
os.makedirs( path )

print("路径被创建")