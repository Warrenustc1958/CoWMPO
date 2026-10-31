"""CoWM-PO SDK: static check for Agent-written world programs (whitelist AST)."""
import ast

SAFE_CALLS = {"range", "len", "min", "max", "abs", "float", "int", "round", "list", "dict", "tuple",
              "enumerate", "zip", "sum"}
ALLOWED_ROOTS = {"w", "math"}
ALLOWED_NODES = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Expr, ast.Constant, ast.Assign,
                 ast.AugAssign, ast.Name, ast.Call, ast.keyword, ast.Attribute, ast.Tuple, ast.List, ast.Dict,
                 ast.BinOp, ast.UnaryOp, ast.For, ast.If, ast.Compare, ast.BoolOp, ast.Return, ast.Subscript,
                 ast.Slice, ast.ListComp, ast.comprehension, ast.IfExp,
                 ast.expr_context, ast.operator, ast.unaryop, ast.cmpop, ast.boolop)


class LintError(ValueError):
    pass


def lint(src):
    tree = ast.parse(src)
    top = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    if len(top) != 1 or not isinstance(top[0], ast.FunctionDef) or top[0].name != "build" \
            or [a.arg for a in top[0].args.args] != ["w"]:
        raise LintError("program must contain exactly one top-level `def build(w):` and nothing else")
    local_names = {"w"}
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise LintError(f"line {getattr(node, 'lineno', '?')}: {type(node).__name__} is not allowed")
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            local_names.add(node.id)
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise LintError(f"line {node.lineno}: private attribute '{node.attr}'")
            if not (isinstance(node.value, ast.Name) and node.value.id in ALLOWED_ROOTS):
                raise LintError(f"line {node.lineno}: attribute access only on w.* or math.*")
        if isinstance(node, ast.Call):
            f = node.func
            ok = (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in ALLOWED_ROOTS) \
                or (isinstance(f, ast.Name) and f.id in SAFE_CALLS)
            if not ok:
                raise LintError(f"line {node.lineno}: call not allowed")
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in local_names | ALLOWED_ROOTS | SAFE_CALLS | {"True", "False", "None"}:
                raise LintError(f"line {node.lineno}: unknown name '{node.id}'")
    return True
