#!/usr/bin/python3
from img_process import Flask_app
from core import host_ip, host_port

if __name__ == '__main__':
    Flask_app.run(host=host_ip, port=host_port)
