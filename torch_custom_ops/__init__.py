from .rag import RagByDragonDelegate


def register_schema_op():
  RagByDragonDelegate.register_schema_op()


def deregister_schema_op():
  RagByDragonDelegate.deregister_schema_op()
