class X:
    
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

    def map(self, function: str):
        for map in self.mapper.items():
            function = function.replace(*map)
        return function
    
    def invmap(self, text: str):
        mapper  = {v: k for k, v in self.mapper.items()}
        for s in mapper.items():
            text = text.replace(*s)
        return text

    def redblue(self, y: tuple, red, blue):
        elems = {
            "y0": y[0],
            "y1": y[1],
            "red": red,
            "blue": blue,
        }
        return self.map("def red-blue([{y0}, {y1}], {red}, {blue}):".format(**elems))
   