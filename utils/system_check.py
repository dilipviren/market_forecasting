import sys

def check_system():
    if sys.platform == 'darwin':
        return 'macOS'
    else:
        return 'win'

if __name__ == "__main__":
    print(check_system())