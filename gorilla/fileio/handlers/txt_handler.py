from .base import BaseFileHandler

class TxtHandler(BaseFileHandler):

    def load_from_fileobj(self, file, **kwargs):
        content = file.readlines()
        return content

    def dump_to_fileobj(self, content, file, **kwargs):
        file.writelines(content)

    def dump_to_str(self, obj, **kwargs):
        pass
