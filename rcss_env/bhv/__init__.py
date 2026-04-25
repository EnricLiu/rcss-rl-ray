from .bhv import BhvMixin
from .neck import BhvNaiveNeck
from .view import BhvHeliosView

class NeckViewBhv:
    __neck: BhvMixin
    __view: BhvMixin

    def __init__(self, neck: BhvMixin = None, view: BhvMixin = None):
        self.__neck = neck or BhvNaiveNeck()
        self.__view = view or BhvHeliosView()

    @property
    def neck(self):
        return self.__neck

    @property
    def view(self):
        return self.__view
