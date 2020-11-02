#! /usr/bin/env python
# -*- coding:utf-8 -*-
import json
import os
import sys

def pretty_json(json_path, insert='pretty'):
    with open(json_path, 'r') as f:
        j = json.load(f)
    json_split_path = os.path.splitext(json_path)
    new_json_path = json_split_path[0] + '_'+ insert + json_split_path[1]
    with open(new_json_path, 'w') as f:
        json.dump(j, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    p = sys.argv[1]
    pretty_json(p)
