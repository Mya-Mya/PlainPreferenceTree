from .pptparser import PPTParser
from .pt import Turn, Subnode, PT

CHANGE_ROLE = {"user": "assistant", "assistant": "user"}
SIGN_TO_SUBNODETYPE = {"+": "upvoted", "-": "downvoted", "*": "writing", "?": "unrated"}


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

        # First turn should be user's
        body = read_body()
        turn = Turn(role="user", main=body)

        def push_turn(turn: Turn) -> Turn:
            pt.append(turn)
            new_role = CHANGE_ROLE[turn.role]
            turn = Turn(role=new_role, main=body)
            return turn

        while lines:
            body = read_body()
            if body == "":
                turn = push_turn(turn)
                continue
            sign = body[0]
            content = body[1:]
            if sign in SIGN_TO_SUBNODETYPE:
                turn.subnodes.append(
                    Subnode(type=SIGN_TO_SUBNODETYPE[sign], content=content)
                )
            else:
                turn = push_turn(turn)
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
