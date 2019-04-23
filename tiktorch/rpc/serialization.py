import zmq
import traceback

from collections import namedtuple
from logging import getLogger
from typing import Any, List, Generic, Iterator, TypeVar, Type, Mapping, Callable, Dict
from zmq.utils import jsonapi

from .types import Cancellation, Signal, MethodCall, MethodReturn, Result


T = TypeVar("T")
logger = getLogger(__name__)


def _exc_to_string(exc):
    exc_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    return "\n".join(exc_lines)


class ISerializer(Generic[T]):
    """
    Serializer interface
    """

    @classmethod
    def deserialize(cls, frames: "FusedFrameIterator") -> T:
        """
        Deserialize objects from zmq.Frame stream
        This function fully consume relevant part of iterator
        """
        raise NotImplementedError

    @classmethod
    def serialize(cls, obj: T) -> Iterator[zmq.Frame]:
        """
        Serialize object to zmq.Frame stream
        """
        raise NotImplementedError


class SerializerRegistry:
    Entry = namedtuple("Entry", ["serializer", "tag"])

    """
    Contains all registered serializers for types
    """

    def __init__(self):
        self._entry_by_type: Dict[Any, self.Entry] = {}
        self._entry_by_tag: Dict[Any, self.Entry] = {}

    def register(self, type_: T, tag: bytes) -> Callable[[ISerializer[T]], Any]:
        """
        Register serializer for given type
        """

        def _reg_fn(cls: ISerializer[T]) -> ISerializer[T]:
            self._entry_by_tag[tag] = self.Entry(cls, tag)
            self._entry_by_type[type_] = self.Entry(cls, tag)
            return cls

        return _reg_fn

    def serialize(self, obj: Any) -> Iterator[zmq.Frame]:
        """
        Serialize single object of type *type_* to zmq.Frame stream
        """
        type_ = type(obj)
        logger.debug("serialize(%r)", type_)
        entry = self._entry_by_type.get(type(obj))

        if not entry:
            raise NotImplementedError(f"Serialization protocol not implemented for {type_}")

        logger.debug("Using %r serializer", entry.serializer)
        yield zmq.Frame(b"T:%s" % entry.tag)

        yield from entry.serializer.serialize(obj)

    def deserialize(self, frames: Iterator[zmq.Frame]) -> Any:
        """
        Deserialize object of type *type_* from frames stream
        """
        logger.debug("deserialize")
        try:
            header_frm = next(frames)
            header = header_frm.bytes
        except StopIteration:
            raise DeserializationError("Failed to read header")

        assert header.startswith(b"T:")
        tag = header[2:]
        entry = self._entry_by_tag.get(tag)
        if not entry:
            raise NotImplementedError(f"Serialization protocol not implemented for {tag}")
        logger.debug("Using %r serializer", entry.serializer)

        return entry.serializer.deserialize(FusedFrameIterator(entry.serializer, frames))


class DeserializationError(Exception):
    pass


class FusedFrameIterator(Iterator[zmq.Frame]):
    """
    Iterate over frames and raise a meaningful error if frame stream is exhausted
    """

    def __init__(self, type_: T, iter_: Iterator[zmq.Frame]) -> None:
        self._iter = iter_
        self._type = type_

    def __iter__(self):
        return self

    def __next__(self) -> zmq.Frame:
        try:
            return next(self._iter)
        except StopIteration:
            raise DeserializationError(f"Not enought frames to deserialize {self._type}") from None


root_reg = SerializerRegistry()
serialize = root_reg.serialize
deserialize = root_reg.deserialize
register = root_reg.register
# TODO Remove
serializer_for = root_reg.register


@serializer_for(type(None), tag=b"none")
class NoneSerializer(ISerializer[None]):
    @classmethod
    def deserialize(cls, frames: FusedFrameIterator) -> None:
        """
        Deserialize objects from zmq.Frame stream
        This function fully consume relevant part of iterator
        """
        frame = next(frames)

        if frame.bytes == b"":
            return None

        raise DeserializationError(f"None frame shouldn't contain any data")

    @classmethod
    def serialize(cls, obj: None) -> Iterator[zmq.Frame]:
        """
        Serialize object to zmq.Frame stream
        """
        yield zmq.Frame()


@serializer_for(bytes, tag=b"bytes")
class BytesSerializer(ISerializer[bytes]):
    @classmethod
    def deserialize(cls, frames: FusedFrameIterator) -> bytes:
        """
        Deserialize objects from zmq.Frame stream
        This function fully consume relevant part of iterator
        """
        frame = next(frames)

        return frame.bytes

    @classmethod
    def serialize(cls, obj: bytes) -> Iterator[zmq.Frame]:
        """
        Serialize object to zmq.Frame stream
        """
        yield zmq.Frame(obj)


@serializer_for(str, tag=b"str")
class StrSerializer(ISerializer[str]):
    @classmethod
    def deserialize(cls, frames: FusedFrameIterator) -> str:
        """
        Deserialize objects from zmq.Frame stream
        This function fully consume relevant part of iterator
        """
        frame = next(frames)

        return frame.bytes.decode("utf-8")

    @classmethod
    def serialize(cls, obj: str) -> Iterator[zmq.Frame]:
        """
        Serialize object to zmq.Frame stream
        """
        yield zmq.Frame(bytes(obj, encoding="utf-8"))


@serializer_for(memoryview, tag=b"mem")
class MemoryViewSerializer(ISerializer[memoryview]):
    @classmethod
    def deserialize(cls, frames: FusedFrameIterator) -> memoryview:
        frame = next(frames)

        return frame.buffer

    @classmethod
    def serialize(cls, obj: memoryview) -> Iterator[zmq.Frame]:
        yield zmq.Frame(obj)


@serializer_for(dict, tag=b"dict")
class DictSerializer(ISerializer[dict]):
    @classmethod
    def serialize(cls, obj: dict) -> Iterator[zmq.Frame]:
        yield zmq.Frame(jsonapi.dumps(obj))

    @classmethod
    def deserialize(cls, frames: FusedFrameIterator) -> dict:
        frame = next(frames)
        return jsonapi.loads(frame.bytes)


@serializer_for(list, tag=b"list")
class ListSerializer(ISerializer[list]):
    @classmethod
    def serialize(cls, obj: list) -> Iterator[zmq.Frame]:
        yield zmq.Frame(jsonapi.dumps(obj))

    @classmethod
    def deserialize(cls, frames: FusedFrameIterator) -> list:
        frame = next(frames)
        return jsonapi.loads(frame.bytes)


@serializer_for(tuple, tag=b"tuple")
class TupleSerializer(ISerializer[tuple]):
    @classmethod
    def serialize(cls, obj: tuple) -> Iterator[zmq.Frame]:
        yield zmq.Frame(jsonapi.dumps(obj))

    @classmethod
    def deserialize(cls, frames: FusedFrameIterator) -> list:
        frame = next(frames)
        return tuple(jsonapi.loads(frame.bytes))


@serializer_for(bool, tag=b"bool")
class BoolSerializer(ISerializer[bool]):
    @classmethod
    def serialize(cls, obj: bool) -> Iterator[zmq.Frame]:
        yield zmq.Frame(b"1" if obj else b"")

    @classmethod
    def deserialize(cls, frames: FusedFrameIterator) -> bool:
        frame = next(frames)
        return bool(frame.bytes)


@serializer_for(tuple, tag=b"tuple")
class TupleSerializer(ISerializer[tuple]):
    @classmethod
    def serialize(cls, obj: tuple) -> Iterator[zmq.Frame]:
        b_len = len(obj).to_bytes(length=2, byteorder="big")
        yield zmq.Frame(b_len)
        for elem in obj:
            yield from serialize(elem)

    @classmethod
    def deserialize(cls, frames: FusedFrameIterator) -> bool:
        len_frame = next(frames)
        len_ = int.from_bytes(len_frame.bytes, byteorder="big")
        return tuple(deserialize(frames) for _ in range(len_))


@serializer_for(MethodCall, tag=b"_call")
class MethodCallSerializer(ISerializer[MethodCall]):
    @classmethod
    def serialize(cls, obj: MethodCall) -> Iterator[zmq.Frame]:
        yield zmq.Frame(obj.id)
        yield zmq.Frame(obj.method_name.encode("utf-8"))
        assert isinstance(obj.args, tuple)
        yield from serialize(obj.args)

    @classmethod
    def deserialize(cls, frames: FusedFrameIterator) -> MethodCall:
        frame_id = next(frames)
        frame_name = next(frames)
        method_name = frame_name.bytes.decode("utf-8")
        args = deserialize(frames)
        return MethodCall(frame_id.bytes, method_name, args, None)


@serializer_for(Result, tag=b"_res")
class ResultSerializer(ISerializer[Result]):
    ERR = b"ERR"
    OK = b"OK"

    @classmethod
    def serialize(cls, obj: Result) -> Iterator[zmq.Frame]:
        if obj.is_err:
            yield zmq.Frame(cls.ERR)
            exc_str = _exc_to_string(obj.error)
            yield zmq.Frame(exc_str.encode("utf-8"))

        else:
            yield zmq.Frame(cls.OK)
            yield from serialize(obj.value)

    @classmethod
    def deserialize(cls, frames: FusedFrameIterator) -> Result:
        marker_frm = next(frames)
        if marker_frm.bytes == cls.OK:
            value = deserialize(frames)
            return Result.OK(value)

        elif marker_frm.bytes == cls.ERR:
            formatted_frm = next(frames)
            decoded = formatted_frm.bytes.decode("utf-8")
            return Result.Error(Exception(decoded))

        else:
            raise DeserializationError("Unknown marker for Result type")


@serializer_for(MethodReturn, tag=b"_ret")
class MethodReturnSerializer(ISerializer[MethodReturn]):
    @classmethod
    def serialize(cls, obj: MethodReturn) -> Iterator[zmq.Frame]:
        yield zmq.Frame(obj.id)
        yield from serialize(obj.result)

    @classmethod
    def deserialize(cls, frames: FusedFrameIterator) -> MethodReturn:
        frame_id = next(frames)
        result = deserialize(frames)
        return MethodReturn(frame_id.bytes, result)


@serializer_for(Cancellation, tag=b"_cancel")
class MethodReturnSerializer(ISerializer[Cancellation]):
    @classmethod
    def serialize(cls, obj: Cancellation) -> Iterator[zmq.Frame]:
        yield zmq.Frame(obj.id)

    @classmethod
    def deserialize(cls, frames: FusedFrameIterator) -> Cancellation:
        frame_id = next(frames)
        return Cancellation(frame_id.bytes)
