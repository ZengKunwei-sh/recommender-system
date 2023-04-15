import os

def path_join(path_list: list) -> str:
    path = path_list[0]
    for i in range(1, len(path_list)):
        path = os.path.join(path, path_list[i])
    return path.replace('\\', '/')

class DataConfig:
    def __init__(self) -> None:
        self.file_path = 'G:/我的云端硬盘/Colab Notebooks/华为视频推荐比赛'
