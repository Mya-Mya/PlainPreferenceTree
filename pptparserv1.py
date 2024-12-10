from .pptparser import PPTParser
from .pt import Turn, Subnode, PT

CHANGE_ROLE = {"user": "assistant", "assistant": "user"}


class PPTParserV1(PPTParser):
    def loads(self, text: str):
        pt: PT = []
        lines = text.splitlines()

        def read_body() -> str:
            first_line = lines.pop(0)
            read_lines = [first_line]
            while lines:
                line = lines[0]
                if line and line[0] == ":":
                    lines.pop(0)
                    read_lines.append(line[1:])
                else:
                    break
            return "\n".join(read_lines)

        # First body should be user's turn
        turn = Turn(role="user", main=read_body())

        while lines:
            body = read_body()
            if body == "":
                body += "\n"
                continue
            sign = body[0]
            content = body[1:]
            if sign == "+":
                turn.subnodes.append(Subnode(type="upvoted", content=content))
            elif sign == "-":
                turn.subnodes.append(Subnode(type="downvoted", content=content))
            elif sign == "*":
                turn.subnodes.append(Subnode(type="writing", content=content))
            elif sign == "?":
                turn.subnodes.append(Subnode(type="unrated", content=content))
            else:
                pt.append(turn)
                new_role = CHANGE_ROLE[turn.role]
                turn = Turn(role=new_role, main=body)
        pt.append(turn)
        return pt

    def dumps(self, pt: PT):
        lines = []

        def put(content: str):
            content_lines = content.splitlines()
            lines.append(content_lines.pop(0))
            for line in content_lines:
                lines.append(":" + line)

        for turn in pt:
            put(turn.main)
            for n in turn.subnodes:
                if n.type == "upvoted":
                    sign = "+"
                if n.type == "downvoted":
                    sign = "-"
                if n.type == "writing":
                    sign = "*"
                if n.type == "unrated":
                    sign = "?"
                put(sign + n.content)

        ppttext = "\n".join(lines)
        return ppttext
