
class X:
    
    """ X is a static class, used as an utility to translate strings from python-like declarations to xent-type for synthetic data generation. You can call its variables directly as X.xdef or define mappings inside of it such as the redblue one. """

    xdef    = "@##$$##@"
    opent     = "(("
    clost     = "))"
    openq     = "[["
    closq     = "]]"
    comma   = "><"
    xreturn = ">:ç%ç>:"
    xend    = ":<ç%ç:<"

    mapper = {
        "(": opent,
        ")": clost,
        "[": openq,
        "]": closq,
        ",": comma,
        ":": xreturn,
        "def": xdef,
        "end": xend
    }

    @classmethod
    def map(cls, function: str):
        for map in cls.mapper.items():
            function = function.replace(*map)
        return function
    
    @classmethod
    def invmap(cls, text: str):
        mapper  = {v: k for k, v in cls.mapper.items()}
        for s in mapper.items():
            text = text.replace(*s)
        return text
    
    @classmethod
    def redblue(cls, y: tuple, red, blue):
        elems = {
            "y0": y[0],
            "y1": y[1],
            "red": red,
            "blue": blue,
        }
        return cls.map("def red-blue([{y0}, {y1}], {red}, {blue}):").format(**elems)
   