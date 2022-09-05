# %%
import imp
import numba
from numba.core import types, cgutils
from numba.extending import make_attribute_wrapper, typeof_impl, models, register_model, unbox, box, NativeValue
import numpy as np
from scipy import sparse


# class csc_matrix:
#     def __init__(self, data: np.array,
#                  indices: np.array,
#                  indptr: np.array,
#                  shape: tuple):
#         self.data = data
#         self.indices = indices
#         self.indptr = indptr
#         self.shape = shape


# class MatrixType(types.Type):
#     def __init__(self, dtype):
#         self.dtype = dtype
#         self.data = types.Array(self.dtype, 1, 'C')
#         self.indices = types.Array(types.int64, 1, 'C')
#         self.indptr = types.Array(types.int64, 1, 'C')
#         self.shape = types.UniTuple(types.int64, 2)
#         self.nnz = types.int64
#         self.ndim = types.int64
#         self.has_sorted_indices = types.boolean
#         super(MatrixType, self).__init__('csc_matrix')


# @typeof_impl.register(csc_matrix)
# def typeof_matrix(val, c):
#     data = typeof_impl(val.data, c)
#     return MatrixType(data.dtype)


# @register_model(MatrixType)
# class MatrixModel(models.StructModel):
#     def __init__(self, dmm, fe_type):
#         members = [
#             ('data', fe_type.data),
#             ('indices', fe_type.indices),
#             ('indptr', fe_type.indptr),
#             ('shape', fe_type.shape)
#         ]
#         models.StructModel.__init__(self, dmm, fe_type, members)


# make_attribute_wrapper(MatrixType, 'data', 'data')
# make_attribute_wrapper(MatrixType, 'indices', 'indices')
# make_attribute_wrapper(MatrixType, 'indptr', 'indptr')
# make_attribute_wrapper(MatrixType, 'shape', 'shape')


# def make_matrix(context, builder, typ, **kwargs):
#     return cgutils.create_struct_proxy(typ)(context,
#                                             builder, **kwargs)


# @unbox(MatrixType)
# def unbox_matrix(typ, obj, c):
#     data = c.pyapi.object_getattr_string(obj, "data")
#     indices = c.pyapi.object_getattr_string(obj, "indices")
#     indptr = c.pyapi.object_getattr_string(obj, "indptr")
#     shape = c.pyapi.object_getattr_string(obj, "shape")
#     matrix = make_matrix(c.context, c.builder, typ)
#     matrix.data = c.unbox(typ.data, data).value
#     matrix.indices = c.unbox(typ.indices, indices).value
#     matrix.indptr = c.unbox(typ.indptr, indptr).value
#     matrix.shape = c.unbox(typ.shape, shape).value
#     for att in [data, indices, indptr, shape]:
#         c.pyapi.decref(att)
#     is_error = cgutils.is_not_null(
#         c.builder, c.pyapi.err_occurred())

#     return NativeValue(matrix._getvalue(), is_error=is_error)


# @box(MatrixType)
# def box_matrix(typ, val, c):
#     matrix = make_matrix(c.context, c.builder, typ)
#     classobj = c.pyapi.unserialize(
#         c.pyapi.serialize_object(csc_matrix))
#     data_obj = c.box(typ.data, matrix.data)
#     indices_obj = c.box(typ.indices, matrix.indices)
#     indptr_obj = c.box(typ.indptr, matrix.indptr)
#     shape_obj = c.box(typ.shape, matrix.shape)
#     matrix_obj = c.pyapi.call_function_objargs(classobj, (
#         data_obj, indices_obj, indptr_obj, shape_obj))
#     return matrix_obj

# @numba.jit(nopython=True)
# def numba_csc_ndarray_dot2(a: csc_matrix, b: np.ndarray):
#     out = np.zeros((a.shape[0], b.shape[1]))
#         for j in range(b.shape[1]):
#             for i in range(b.shape[0]):
#        for k in range(a.indptr[i], a.indptr[i + 1]):
#       out[a.indices[k], j] += a.data[k] * b[i, j]
#     return out

@numba.jit(nopython=True)
def numba_csc_ndarray_dot2(data: np.ndarray, indices: np.ndarray, indptr: np.ndarray,
                           shape: np.ndarray, b: np.ndarray):
    out = np.zeros((shape[0], b.shape[1]))
    for j in range(b.shape[1]):
        for i in range(b.shape[0]):
            for k in range(indptr[i], indptr[i + 1]):
                out[indices[k], j] += data[k] * b[i, j]
    return out


# %%
if __name__ == '__main__':
    a = sparse.csc_matrix((np.array([1, 2, 3, 4, 5, 6]), np.array([0, 1, 2, 0, 1, 2]),
                           np.array([0, 2, 3, 6])), (3, 3))
    A = sparse.csc_matrix(a.data[:], a.indices[:], a.indptr, a.shape[:])
    b = np.array([[1], [2], [3]])

    print(numba_csc_ndarray_dot2(a.data, a.indices, a.indptr, a.shape, b))
# %%
