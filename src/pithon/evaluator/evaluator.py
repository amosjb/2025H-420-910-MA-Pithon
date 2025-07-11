from pithon.evaluator.envframe import EnvFrame
from pithon.evaluator.primitive import check_type, get_primitive_dict
from pithon.syntax import (
    PiAssignment, PiBinaryOperation, PiNumber, PiBool, PiStatement, PiProgram, PiSubscript, PiVariable,
    PiIfThenElse, PiNot, PiAnd, PiOr, PiWhile, PiNone, PiList, PiTuple, PiString,
    PiFunctionDef, PiFunctionCall, PiFor, PiBreak, PiContinue, PiIn, PiReturn, PiClassDef, PiAttribute, PiAttributeAssignment
)
from pithon.evaluator.envvalue import (
    EnvValue, VFunctionClosure, VList, VNone, VTuple, VNumber, VBool, VString, VClassDef, VObject, VMethodClosure
)

def initial_env() -> EnvFrame:
    """Crée et retourne l'environnement initial avec les primitives."""
    env = EnvFrame()
    env.vars.update(get_primitive_dict())
    return env

def lookup(env: EnvFrame, name: str) -> EnvValue:
    """Recherche une variable dans l'environnement."""
    return env.lookup(name)

def insert(env: EnvFrame, name: str, value: EnvValue) -> None:
    """Insère une variable dans l'environnement."""
    env.insert(name, value)

def evaluate(node: PiProgram, env: EnvFrame) -> EnvValue:
    """Évalue un programme ou une liste d'instructions."""
    try:
        if isinstance(node, list):
            last_value = VNone(value=None)
            for stmt in node:
                last_value = evaluate_stmt(stmt, env)
            return last_value
        elif isinstance(node, PiStatement):
            return evaluate_stmt(node, env)
        else:
            raise TypeError(f"Type de nœud non supporté : {type(node)}")
    except (TypeError, ZeroDivisionError, ValueError) as e:
        raise RuntimeError(str(e))

def evaluate_stmt(node: PiStatement, env: EnvFrame) -> EnvValue:
    """Évalue une instruction ou expression Pithon."""
    try:
        if isinstance(node, PiNumber):
            return VNumber(node.value)
    
        elif isinstance(node, PiBool):
            return VBool(node.value)
    
        elif isinstance(node, PiNone):
            return VNone(node.value)
    
        elif isinstance(node, PiString):
            return VString(node.value)
    
        elif isinstance(node, PiList):
            elements = [evaluate_stmt(e, env) for e in node.elements]
            return VList(elements)
    
        elif isinstance(node, PiTuple):
            elements = tuple(evaluate_stmt(e, env) for e in node.elements)
            return VTuple(elements)
    
        elif isinstance(node, PiVariable):
            return lookup(env, node.name)
    
        elif isinstance(node, PiBinaryOperation):
            # Traite l'opération binaire comme un appel de fonction
            fct_call = PiFunctionCall(
                function=PiVariable(name=node.operator),
                args=[node.left, node.right]
            )
            return evaluate_stmt(fct_call, env)
    
        elif isinstance(node, PiAssignment):
            value = evaluate_stmt(node.value, env)
            insert(env, node.name, value)
            return value
    
        elif isinstance(node, PiIfThenElse):
            cond = evaluate_stmt(node.condition, env)
            cond = check_type(cond, VBool)
            branch = node.then_branch if cond.value else node.else_branch
            last_value = evaluate(branch, env)
            return last_value
    
        elif isinstance(node, PiNot):
            operand = evaluate_stmt(node.operand, env)
            # Vérifie le type pour l'opérateur 'not'
            _check_valid_piandor_type(operand)
            return VBool(not operand.value) # type: ignore
    
        elif isinstance(node, PiAnd):
            left = evaluate_stmt(node.left, env)
            _check_valid_piandor_type(left)
            if not left.value: # type: ignore
                return left
            right = evaluate_stmt(node.right, env)
            _check_valid_piandor_type(right)
            return right
    
        elif isinstance(node, PiOr):
            left = evaluate_stmt(node.left, env)
            _check_valid_piandor_type(left)
            if left.value: # type: ignore
                return left
            right = evaluate_stmt(node.right, env)
            _check_valid_piandor_type(right)
            return right
    
        elif isinstance(node, PiWhile):
            return _evaluate_while(node, env)
    
        elif isinstance(node, PiFunctionDef):
            closure = VFunctionClosure(node, env)
            insert(env, node.name, closure)
            return VNone(value=None)
    
        elif isinstance(node, PiReturn):
            value = evaluate_stmt(node.value, env)
            raise ReturnException(value)
    
        elif isinstance(node, PiFunctionCall):
            return _evaluate_function_call(node, env)
    
        elif isinstance(node, PiFor):
            return _evaluate_for(node, env)
    
        elif isinstance(node, PiBreak):
            raise BreakException()
    
        elif isinstance(node, PiContinue):
            raise ContinueException()
    
        elif isinstance(node, PiIn):
            return _evaluate_in(node, env)
    
        elif isinstance(node, PiSubscript):
            return _evaluate_subscript(node, env)

        elif isinstance(node, PiClassDef):
            methods = {
                method.name: VFunctionClosure(method, env)
                for method in node.methods
            }
            class_def = VClassDef(name=node.name, methods=methods)
            env.insert(node.name, class_def)
            return VNone(value=None)

        elif isinstance(node, PiAttribute):
            obj = evaluate_stmt(node.object, env)
            if not isinstance(obj, VObject):
                raise TypeError("Seuls les objets peuvent avoir des attributs.")
            if node.attr in obj.attributes:
                return obj.attributes[node.attr]
            if node.attr in obj.class_def.methods:
                method = obj.class_def.methods[node.attr]
                return VMethodClosure(function=method, instance=obj)
            raise AttributeError(f"L'objet n'a pas d'attribut '{node.attr}'")

        elif isinstance(node, PiAttributeAssignment):
            obj = evaluate_stmt(node.object, env)
            if not isinstance(obj, VObject):
                raise TypeError("Seuls les objets peuvent avoir des attributs.")
            value = evaluate_stmt(node.value, env)
            obj.attributes[node.attr] = value
            return value
    
        else:
            raise TypeError(f"Type de nœud non supporté : {type(node)}")
    except (TypeError, ZeroDivisionError, ValueError) as e:
        raise RuntimeError(str(e))

def _check_valid_piandor_type(obj):
    """Vérifie que le type est valide pour 'and'/'or'."""
    if not isinstance(obj, VBool | VNumber | VString | VNone | VList | VTuple):
        raise TypeError(f"Type non supporté pour l'opérateur 'and': {type(obj).__name__}")

def _evaluate_while(node: PiWhile, env: EnvFrame) -> EnvValue:
    """Évalue une boucle while."""
    last_value = VNone(value=None)
    while True:
        cond = evaluate_stmt(node.condition, env)
        cond = check_type(cond, VBool)
        if not cond.value:
            break
        try:
            last_value = evaluate(node.body, env)
        except BreakException:
            break
        except ContinueException:
            continue
    return last_value

def _evaluate_for(node: PiFor, env: EnvFrame) -> EnvValue:
    """Évalue une boucle for."""
    iterable_val = evaluate_stmt(node.iterable, env)
    if not isinstance(iterable_val, (VList, VTuple)):
        raise TypeError("La boucle for attend une liste ou un tuple.")
    last_value = VNone(value=None)
    iterable = iterable_val.value
    for item in iterable:
        env.insert(node.var, item)  # Pas de nouvel environnement pour la variable de boucle
        try:
            last_value = evaluate(node.body, env)
        except BreakException:
            break
        except ContinueException:
            continue
    return last_value

def _evaluate_subscript(node: PiSubscript, env: EnvFrame) -> EnvValue:
    """Évalue une opération d'indexation (subscript)."""
    collection = evaluate_stmt(node.collection, env)
    index = evaluate_stmt(node.index, env)
    # Indexation pour liste, tuple ou chaîne
    if isinstance(collection, VList):
        idx = check_type(index, VNumber)
        return collection.value[int(idx.value)]
    elif isinstance(collection, VTuple):
        idx = check_type(index, VNumber)
        return collection.value[int(idx.value)]
    elif isinstance(collection, VString):
        idx = check_type(index, VNumber)
        return VString(collection.value[int(idx.value)])
    else:
        raise TypeError("L'indexation n'est supportée que pour les listes, tuples et chaînes.")

def _evaluate_in(node: PiIn, env: EnvFrame) -> EnvValue:
    """Évalue l'opérateur 'in'."""
    container = evaluate_stmt(node.container, env)
    element = evaluate_stmt(node.element, env)
    if isinstance(container, (VList, VTuple)):
        return VBool(element in container.value)
    elif isinstance(container, VString):
        if isinstance(element, VString):
            return VBool(element.value in container.value)
        else:
            return VBool(False)
    else:
        raise TypeError("'in' n'est supporté que pour les listes et chaînes.")

def _evaluate_function_call(node: PiFunctionCall, env: EnvFrame) -> EnvValue:
    """Évalue un appel de fonction (primitive ou définie par l'utilisateur)."""
    func_val = evaluate_stmt(node.function, env)
    args = [evaluate_stmt(arg, env) for arg in node.args]

    # Instanciation de classe
    if isinstance(func_val, VClassDef):
        instance = VObject(class_def=func_val, attributes={})
        # Chercher et appeler la méthode __init__ si elle existe
        if '__init__' in func_val.methods:
            init_method = func_val.methods['__init__']
            init_closure = VMethodClosure(function=init_method, instance=instance)
            _execute_method_call(init_closure, args, env)
        return instance

    # Appel de méthode
    if isinstance(func_val, VMethodClosure):
        return _execute_method_call(func_val, args, env)
    
    # Fonction primitive
    if callable(func_val):
        return func_val(args)
    # Fonction utilisateur
    if not isinstance(func_val, VFunctionClosure):
        raise TypeError("Tentative d'appel d'un objet non-fonction.")
    return _execute_function_call(func_val, args, env)

def _execute_function_call(func_val: VFunctionClosure, args: list[EnvValue], env: EnvFrame) -> EnvValue:
    funcdef = func_val.funcdef
    closure_env = func_val.closure_env
    call_env = EnvFrame(parent=closure_env)
    for i, arg_name in enumerate(funcdef.arg_names):
        if i < len(args):
            call_env.insert(arg_name, args[i])
        else:
            raise TypeError("Argument manquant pour la fonction.")
    if funcdef.vararg:
        varargs = VList(args[len(funcdef.arg_names):])
        call_env.insert(funcdef.vararg, varargs)
    elif len(args) > len(funcdef.arg_names):
        raise TypeError("Trop d'arguments pour la fonction.")
    result = VNone(value=None)
    try:
        for stmt in funcdef.body:
            result = evaluate_stmt(stmt, call_env)
    except ReturnException as ret:
        return ret.value
    return result

def _execute_method_call(method_closure: VMethodClosure, args: list[EnvValue], env: EnvFrame) -> EnvValue:
    """Exécute un appel de méthode."""
    funcdef = method_closure.function.funcdef
    closure_env = method_closure.function.closure_env
    call_env = EnvFrame(parent=closure_env)
    
    # 'self' est le premier argument
    if not funcdef.arg_names or funcdef.arg_names[0] != 'self':
        raise TypeError("La première argument d'une méthode doit être 'self'.")
    
    call_env.insert('self', method_closure.instance)
    
    # Les autres arguments
    arg_names = funcdef.arg_names[1:]
    for i, arg_name in enumerate(arg_names):
        if i < len(args):
            call_env.insert(arg_name, args[i])
        else:
            raise TypeError(f"Argument manquant '{arg_name}' pour la méthode.")

    if len(args) > len(arg_names):
        raise TypeError("Trop d'arguments pour la méthode.")

    result = VNone(value=None)
    try:
        for stmt in funcdef.body:
            result = evaluate_stmt(stmt, call_env)
    except ReturnException as ret:
        return ret.value
    
    # Pour __init__, on retourne None implicitement
    if funcdef.name == '__init__':
        return VNone(value=None)
        
    return result

class ReturnException(Exception):
    """Exception pour retourner une valeur depuis une fonction."""
    def __init__(self, value):
        self.value = value

class BreakException(Exception):
    """Exception pour sortir d'une boucle (break)."""
    pass

class ContinueException(Exception):
    """Exception pour passer à l'itération suivante (continue)."""
    pass
