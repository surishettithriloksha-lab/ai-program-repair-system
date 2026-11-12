
import ast
import joblib
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix, hstack

# Load trained model
saved = joblib.load("model.joblib")
model = saved["model"]
tfidf = saved["tfidf"]
num_cols = saved["num_cols"]

NODE_TYPES = [
    "If", "For", "While", "BoolOp", "Compare", "BinOp",
    "Subscript", "Call", "Return", "Assign", "AugAssign",
    "Try", "ExceptHandler", "ListComp", "Lambda",
    "Break", "Continue"
]

CMP = [
    "Lt", "LtE", "Gt", "GtE", "Eq", "NotEq",
    "Is", "IsNot", "In", "NotIn"
]

BIN = [
    "Add", "Sub", "Mult", "Div", "FloorDiv", "Mod",
    "BitAnd", "BitOr", "BitXor", "Pow"
]


def depth(node):
    kids = list(ast.iter_child_nodes(node))
    return 1 + max((depth(k) for k in kids), default=0)


def has_len_call(node):
    return any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "len"
        for n in ast.walk(node)
    )


def extract(code):
    tree = ast.parse(code)
    nodes = list(ast.walk(tree))
    f = {}

    for t in NODE_TYPES:
        f["n_" + t] = sum(type(n).__name__ == t for n in nodes)

    for t in CMP:
        f["cmp_" + t] = sum(
            type(op).__name__ == t
            for n in nodes if isinstance(n, ast.Compare)
            for op in n.ops
        )

    for t in BIN:
        f["bin_" + t] = sum(
            isinstance(n, ast.BinOp) and type(n.op).__name__ == t
            for n in nodes
        )

    f["loc"] = len(code.splitlines())
    f["depth"] = depth(tree)
    f["n_nodes"] = len(nodes)

    f["cyclomatic"] = 1 + sum(
        isinstance(n, (ast.If, ast.For, ast.While,
                       ast.BoolOp, ast.ExceptHandler))
        for n in nodes
    )

    fn = tree.body[0]
    f["n_params"] = (
        len(fn.args.args) if isinstance(fn, ast.FunctionDef) else 0
    )

    f["sub_with_binop"] = sum(
        isinstance(n, ast.Subscript)
        and isinstance(n.slice, ast.BinOp) for n in nodes
    )

    f["range_with_binop"] = sum(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "range"
        and any(isinstance(a, ast.BinOp) for a in n.args)
        for n in nodes
    )

    f["n_len_calls"] = sum(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "len" for n in nodes
    )

    f["cmp_with_len"] = sum(
        isinstance(n, ast.Compare) and has_len_call(n)
        for n in nodes
    )

    f["n_numeric_consts"] = sum(
        isinstance(n, ast.Constant)
        and isinstance(n.value, (int, float)) for n in nodes
    )

    f["n_division"] = sum(
        isinstance(n, ast.BinOp)
        and isinstance(n.op, (ast.Div, ast.FloorDiv, ast.Mod))
        for n in nodes
    )

    return f


def suspicious_lines(code):
    tree = ast.parse(code)
    found = {}

    for node in ast.walk(tree):
        if not hasattr(node, "lineno"):
            continue

        reason = None

        if isinstance(node, ast.Subscript):
            reason = "Array or list indexing: check index boundaries"

        elif isinstance(node, ast.Compare) and has_len_call(node):
            reason = "Length comparison: check boundary conditions"

        elif isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Div, ast.FloorDiv, ast.Mod)
        ):
            reason = "Arithmetic operation: verify edge cases"

        elif isinstance(node, (ast.For, ast.While)):
            reason = "Loop: check termination and iteration boundaries"

        if reason:
            found[node.lineno] = reason

    return found


# Dashboard design
st.set_page_config(
    page_title="AI Program Repair System",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI-Powered Automated Program Repair System")
st.caption(
    "Intelligent software defect prediction and code analysis"
)

st.divider()

st.subheader("1. Enter Python Code")

code = st.text_area(
    "Paste your Python function below:",
    height=250,
    placeholder="def example(arr):\n    for i in range(len(arr)):\n        print(arr[i])"
)

if st.button("🔍 Analyse Code", type="primary"):

    if not code.strip():
        st.warning("Please enter some Python code.")

    else:
        try:
            ast.parse(code)

            features = extract(code)

            num = pd.DataFrame([features]).reindex(
                columns=num_cols, fill_value=0
            )

            tf = tfidf.transform([code])

            X = hstack([
                csr_matrix(num.values.astype(float)),
                tf
            ]).tocsr()

            probability = model.predict_proba(X)[0]

            positive_index = list(model.classes_).index(1)
            bug_probability = probability[positive_index]

            prediction = int(bug_probability >= 0.5)

            st.divider()
            st.subheader("2. Defect Prediction")

            col1, col2 = st.columns(2)

            with col1:
                if prediction == 1:
                    st.error("Potential Buggy Code Detected")
                else:
                    st.success("No Defect Predicted")

            with col2:
                st.metric(
                    "Bug Probability",
                    f"{bug_probability * 100:.2f}%"
                )

            st.progress(float(bug_probability))

            st.caption(
                "This is a model probability, not a guarantee "
                "that the code is correct or defective."
            )

            st.divider()
            st.subheader("3. Code Analysis")

            c1, c2, c3 = st.columns(3)

            c1.metric("Lines of Code", len(code.splitlines()))
            c2.metric("AST Nodes", sum(features.values()) if False else features["n_nodes"])
            c3.metric("Cyclomatic Complexity", features["cyclomatic"])

            st.divider()
            st.subheader("4. Suspicious Code Localization")

            locations = suspicious_lines(code)

            if locations:
                for line, reason in sorted(locations.items()):
                    st.warning(f"Line {line}: {reason}")
                    st.code(
                        code.splitlines()[line - 1],
                        language="python"
                    )
            else:
                st.info("No suspicious patterns found by the basic rule checks.")

            st.divider()
            st.subheader("5. Analysis Summary")

            st.write(
                "The Random Forest model analysed the submitted "
                "code using structural and textual features."
            )

            st.write(
                "The highlighted lines are rule-based review candidates. "
                "They are not confirmed defects or automatic repairs."
            )

        except SyntaxError as e:
            st.error(f"Python syntax error: {e}")

        except Exception as e:
            st.error(f"Analysis error: {e}")

st.divider()
st.caption("AI Program Repair System | Hackathon Project")
