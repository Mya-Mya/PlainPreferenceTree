from .pptparser import PPTParser
from PlainPreferenceTree.pt import Turn, Subnode, PT

NEXT_ROLE = {"system":"user", "user": "assistant", "assistant": "user"}
SUBNODE_SIGN_TO_NAME = {"+": "upvoted", "-": "downvoted", "*": "writing", "?": "unrated"}

class PPTParserV2(PPTParser):
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

        # First turn
        body = read_body()
        role = "user"
        if body and body[0] == "/":
            body = body[1:]
            role = "system"
        turn = Turn(role=role, main=body)

        def push_turn(turn: Turn) -> Turn:
            pt.append(turn)
            new_role = NEXT_ROLE[turn.role]
            turn = Turn(role=new_role, main=body)
            return turn

        while lines:
            body = read_body()
            if body == "":
                turn = push_turn(turn)
                continue
            sign = body[0]
            content = body[1:]
            if sign in SUBNODE_SIGN_TO_NAME:
                turn.subnodes.append(
                    Subnode(type=SUBNODE_SIGN_TO_NAME[sign], content=content)
                )
            else:
                turn = push_turn(turn)
        pt.append(turn)
        return pt

    def dumps(self, pt: PT):
        lines = []

        def put(content: str):
            content_lines = content.splitlines()
            if not content_lines:
                lines.append("")
                return
            lines.append(content_lines.pop(0))
            for line in content_lines:
                lines.append(":" + line)

        for turn in pt:
            main_content = turn.main
            if turn.role == "system":
                main_content = "/" + main_content
            put(main_content)
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
