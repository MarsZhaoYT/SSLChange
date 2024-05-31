import argparse as ag
import json


def get_parser_with_args(metadata_json='Transfer-Model/metadata.json'):
    # 定义一个参数解析器parser
    parser = ag.ArgumentParser(description='Training change detection network')

    with open(metadata_json, 'r') as fin:
        metadata = json.load(fin)   # 读取metadata.json中存储的字符串并转为字典对象
        parser.set_defaults(**metadata)
        return parser, metadata


def print_options(opt):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in vars(opt).items():
        message += '{:>20}: {:<30}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
    print(message)