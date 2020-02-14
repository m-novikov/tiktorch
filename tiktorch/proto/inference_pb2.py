# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: inference.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='inference.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x0finference.proto\"Y\n\x06\x44\x65vice\x12\n\n\x02id\x18\x01 \x01(\t\x12\x1e\n\x06status\x18\x02 \x01(\x0e\x32\x0e.Device.Status\"#\n\x06Status\x12\r\n\tAVAILABLE\x10\x00\x12\n\n\x06IN_USE\x10\x01\"\'\n\x04\x42lob\x12\x0e\n\x06\x66ormat\x18\x01 \x01(\t\x12\x0f\n\x07\x63ontent\x18\x02 \x01(\x0c\"i\n\x19\x43reateModelSessionRequest\x12\x13\n\tmodel_uri\x18\x01 \x01(\tH\x00\x12\x1b\n\nmodel_blob\x18\x02 \x01(\x0b\x32\x05.BlobH\x00\x12\x11\n\tdeviceIds\x18\x05 \x03(\tB\x07\n\x05model\"(\n\x0cModelSession\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0c\n\x04name\x18\x02 \x01(\t\"\x9e\x01\n\x08LogEntry\x12\x11\n\ttimestamp\x18\x01 \x01(\r\x12\x1e\n\x05level\x18\x02 \x01(\x0e\x32\x0f.LogEntry.Level\x12\x0f\n\x07\x63ontent\x18\x03 \x01(\t\"N\n\x05Level\x12\n\n\x06NOTSET\x10\x00\x12\t\n\x05\x44\x45\x42UG\x10\x01\x12\x08\n\x04INFO\x10\x02\x12\x0b\n\x07WARNING\x10\x03\x12\t\n\x05\x45RROR\x10\x04\x12\x0c\n\x08\x43RITICAL\x10\x05\"#\n\x07\x44\x65vices\x12\x18\n\x07\x64\x65vices\x18\x01 \x03(\x0b\x32\x07.Device\"\'\n\tTensorDim\x12\x0c\n\x04size\x18\x01 \x01(\r\x12\x0c\n\x04name\x18\x02 \x01(\t\"B\n\x06Tensor\x12\x0e\n\x06\x62uffer\x18\x01 \x01(\x0c\x12\r\n\x05\x64type\x18\x02 \x01(\t\x12\x19\n\x05shape\x18\x03 \x03(\x0b\x32\n.TensorDim\"A\n\x0ePredictRequest\x12\x16\n\x0emodelSessionId\x18\x01 \x01(\t\x12\x17\n\x06tensor\x18\x02 \x01(\x0b\x32\x07.Tensor\"*\n\x0fPredictResponse\x12\x17\n\x06tensor\x18\x01 \x01(\x0b\x32\x07.Tensor\"\x07\n\x05\x45mpty2\xf1\x01\n\tInference\x12\x41\n\x12\x43reateModelSession\x12\x1a.CreateModelSessionRequest\x1a\r.ModelSession\"\x00\x12,\n\x11\x43loseModelSession\x12\r.ModelSession\x1a\x06.Empty\"\x00\x12 \n\x07GetLogs\x12\x06.Empty\x1a\t.LogEntry\"\x00\x30\x01\x12!\n\x0bListDevices\x12\x06.Empty\x1a\x08.Devices\"\x00\x12.\n\x07Predict\x12\x0f.PredictRequest\x1a\x10.PredictResponse\"\x00\x32G\n\rFlightControl\x12\x18\n\x04Ping\x12\x06.Empty\x1a\x06.Empty\"\x00\x12\x1c\n\x08Shutdown\x12\x06.Empty\x1a\x06.Empty\"\x00\x62\x06proto3')
)



_DEVICE_STATUS = _descriptor.EnumDescriptor(
  name='Status',
  full_name='Device.Status',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='AVAILABLE', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='IN_USE', index=1, number=1,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=73,
  serialized_end=108,
)
_sym_db.RegisterEnumDescriptor(_DEVICE_STATUS)

_LOGENTRY_LEVEL = _descriptor.EnumDescriptor(
  name='Level',
  full_name='LogEntry.Level',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='NOTSET', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DEBUG', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='INFO', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='WARNING', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ERROR', index=4, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CRITICAL', index=5, number=5,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=381,
  serialized_end=459,
)
_sym_db.RegisterEnumDescriptor(_LOGENTRY_LEVEL)


_DEVICE = _descriptor.Descriptor(
  name='Device',
  full_name='Device',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='Device.id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='status', full_name='Device.status', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _DEVICE_STATUS,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=19,
  serialized_end=108,
)


_BLOB = _descriptor.Descriptor(
  name='Blob',
  full_name='Blob',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='format', full_name='Blob.format', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='content', full_name='Blob.content', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=110,
  serialized_end=149,
)


_CREATEMODELSESSIONREQUEST = _descriptor.Descriptor(
  name='CreateModelSessionRequest',
  full_name='CreateModelSessionRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='model_uri', full_name='CreateModelSessionRequest.model_uri', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='model_blob', full_name='CreateModelSessionRequest.model_blob', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='deviceIds', full_name='CreateModelSessionRequest.deviceIds', index=2,
      number=5, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='model', full_name='CreateModelSessionRequest.model',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=151,
  serialized_end=256,
)


_MODELSESSION = _descriptor.Descriptor(
  name='ModelSession',
  full_name='ModelSession',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='ModelSession.id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='name', full_name='ModelSession.name', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=258,
  serialized_end=298,
)


_LOGENTRY = _descriptor.Descriptor(
  name='LogEntry',
  full_name='LogEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='timestamp', full_name='LogEntry.timestamp', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='level', full_name='LogEntry.level', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='content', full_name='LogEntry.content', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _LOGENTRY_LEVEL,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=301,
  serialized_end=459,
)


_DEVICES = _descriptor.Descriptor(
  name='Devices',
  full_name='Devices',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='devices', full_name='Devices.devices', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=461,
  serialized_end=496,
)


_TENSORDIM = _descriptor.Descriptor(
  name='TensorDim',
  full_name='TensorDim',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='size', full_name='TensorDim.size', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='name', full_name='TensorDim.name', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=498,
  serialized_end=537,
)


_TENSOR = _descriptor.Descriptor(
  name='Tensor',
  full_name='Tensor',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='buffer', full_name='Tensor.buffer', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dtype', full_name='Tensor.dtype', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='shape', full_name='Tensor.shape', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=539,
  serialized_end=605,
)


_PREDICTREQUEST = _descriptor.Descriptor(
  name='PredictRequest',
  full_name='PredictRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='modelSessionId', full_name='PredictRequest.modelSessionId', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tensor', full_name='PredictRequest.tensor', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=607,
  serialized_end=672,
)


_PREDICTRESPONSE = _descriptor.Descriptor(
  name='PredictResponse',
  full_name='PredictResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='tensor', full_name='PredictResponse.tensor', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=674,
  serialized_end=716,
)


_EMPTY = _descriptor.Descriptor(
  name='Empty',
  full_name='Empty',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=718,
  serialized_end=725,
)

_DEVICE.fields_by_name['status'].enum_type = _DEVICE_STATUS
_DEVICE_STATUS.containing_type = _DEVICE
_CREATEMODELSESSIONREQUEST.fields_by_name['model_blob'].message_type = _BLOB
_CREATEMODELSESSIONREQUEST.oneofs_by_name['model'].fields.append(
  _CREATEMODELSESSIONREQUEST.fields_by_name['model_uri'])
_CREATEMODELSESSIONREQUEST.fields_by_name['model_uri'].containing_oneof = _CREATEMODELSESSIONREQUEST.oneofs_by_name['model']
_CREATEMODELSESSIONREQUEST.oneofs_by_name['model'].fields.append(
  _CREATEMODELSESSIONREQUEST.fields_by_name['model_blob'])
_CREATEMODELSESSIONREQUEST.fields_by_name['model_blob'].containing_oneof = _CREATEMODELSESSIONREQUEST.oneofs_by_name['model']
_LOGENTRY.fields_by_name['level'].enum_type = _LOGENTRY_LEVEL
_LOGENTRY_LEVEL.containing_type = _LOGENTRY
_DEVICES.fields_by_name['devices'].message_type = _DEVICE
_TENSOR.fields_by_name['shape'].message_type = _TENSORDIM
_PREDICTREQUEST.fields_by_name['tensor'].message_type = _TENSOR
_PREDICTRESPONSE.fields_by_name['tensor'].message_type = _TENSOR
DESCRIPTOR.message_types_by_name['Device'] = _DEVICE
DESCRIPTOR.message_types_by_name['Blob'] = _BLOB
DESCRIPTOR.message_types_by_name['CreateModelSessionRequest'] = _CREATEMODELSESSIONREQUEST
DESCRIPTOR.message_types_by_name['ModelSession'] = _MODELSESSION
DESCRIPTOR.message_types_by_name['LogEntry'] = _LOGENTRY
DESCRIPTOR.message_types_by_name['Devices'] = _DEVICES
DESCRIPTOR.message_types_by_name['TensorDim'] = _TENSORDIM
DESCRIPTOR.message_types_by_name['Tensor'] = _TENSOR
DESCRIPTOR.message_types_by_name['PredictRequest'] = _PREDICTREQUEST
DESCRIPTOR.message_types_by_name['PredictResponse'] = _PREDICTRESPONSE
DESCRIPTOR.message_types_by_name['Empty'] = _EMPTY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Device = _reflection.GeneratedProtocolMessageType('Device', (_message.Message,), dict(
  DESCRIPTOR = _DEVICE,
  __module__ = 'inference_pb2'
  # @@protoc_insertion_point(class_scope:Device)
  ))
_sym_db.RegisterMessage(Device)

Blob = _reflection.GeneratedProtocolMessageType('Blob', (_message.Message,), dict(
  DESCRIPTOR = _BLOB,
  __module__ = 'inference_pb2'
  # @@protoc_insertion_point(class_scope:Blob)
  ))
_sym_db.RegisterMessage(Blob)

CreateModelSessionRequest = _reflection.GeneratedProtocolMessageType('CreateModelSessionRequest', (_message.Message,), dict(
  DESCRIPTOR = _CREATEMODELSESSIONREQUEST,
  __module__ = 'inference_pb2'
  # @@protoc_insertion_point(class_scope:CreateModelSessionRequest)
  ))
_sym_db.RegisterMessage(CreateModelSessionRequest)

ModelSession = _reflection.GeneratedProtocolMessageType('ModelSession', (_message.Message,), dict(
  DESCRIPTOR = _MODELSESSION,
  __module__ = 'inference_pb2'
  # @@protoc_insertion_point(class_scope:ModelSession)
  ))
_sym_db.RegisterMessage(ModelSession)

LogEntry = _reflection.GeneratedProtocolMessageType('LogEntry', (_message.Message,), dict(
  DESCRIPTOR = _LOGENTRY,
  __module__ = 'inference_pb2'
  # @@protoc_insertion_point(class_scope:LogEntry)
  ))
_sym_db.RegisterMessage(LogEntry)

Devices = _reflection.GeneratedProtocolMessageType('Devices', (_message.Message,), dict(
  DESCRIPTOR = _DEVICES,
  __module__ = 'inference_pb2'
  # @@protoc_insertion_point(class_scope:Devices)
  ))
_sym_db.RegisterMessage(Devices)

TensorDim = _reflection.GeneratedProtocolMessageType('TensorDim', (_message.Message,), dict(
  DESCRIPTOR = _TENSORDIM,
  __module__ = 'inference_pb2'
  # @@protoc_insertion_point(class_scope:TensorDim)
  ))
_sym_db.RegisterMessage(TensorDim)

Tensor = _reflection.GeneratedProtocolMessageType('Tensor', (_message.Message,), dict(
  DESCRIPTOR = _TENSOR,
  __module__ = 'inference_pb2'
  # @@protoc_insertion_point(class_scope:Tensor)
  ))
_sym_db.RegisterMessage(Tensor)

PredictRequest = _reflection.GeneratedProtocolMessageType('PredictRequest', (_message.Message,), dict(
  DESCRIPTOR = _PREDICTREQUEST,
  __module__ = 'inference_pb2'
  # @@protoc_insertion_point(class_scope:PredictRequest)
  ))
_sym_db.RegisterMessage(PredictRequest)

PredictResponse = _reflection.GeneratedProtocolMessageType('PredictResponse', (_message.Message,), dict(
  DESCRIPTOR = _PREDICTRESPONSE,
  __module__ = 'inference_pb2'
  # @@protoc_insertion_point(class_scope:PredictResponse)
  ))
_sym_db.RegisterMessage(PredictResponse)

Empty = _reflection.GeneratedProtocolMessageType('Empty', (_message.Message,), dict(
  DESCRIPTOR = _EMPTY,
  __module__ = 'inference_pb2'
  # @@protoc_insertion_point(class_scope:Empty)
  ))
_sym_db.RegisterMessage(Empty)



_INFERENCE = _descriptor.ServiceDescriptor(
  name='Inference',
  full_name='Inference',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=728,
  serialized_end=969,
  methods=[
  _descriptor.MethodDescriptor(
    name='CreateModelSession',
    full_name='Inference.CreateModelSession',
    index=0,
    containing_service=None,
    input_type=_CREATEMODELSESSIONREQUEST,
    output_type=_MODELSESSION,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='CloseModelSession',
    full_name='Inference.CloseModelSession',
    index=1,
    containing_service=None,
    input_type=_MODELSESSION,
    output_type=_EMPTY,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='GetLogs',
    full_name='Inference.GetLogs',
    index=2,
    containing_service=None,
    input_type=_EMPTY,
    output_type=_LOGENTRY,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='ListDevices',
    full_name='Inference.ListDevices',
    index=3,
    containing_service=None,
    input_type=_EMPTY,
    output_type=_DEVICES,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='Predict',
    full_name='Inference.Predict',
    index=4,
    containing_service=None,
    input_type=_PREDICTREQUEST,
    output_type=_PREDICTRESPONSE,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_INFERENCE)

DESCRIPTOR.services_by_name['Inference'] = _INFERENCE


_FLIGHTCONTROL = _descriptor.ServiceDescriptor(
  name='FlightControl',
  full_name='FlightControl',
  file=DESCRIPTOR,
  index=1,
  serialized_options=None,
  serialized_start=971,
  serialized_end=1042,
  methods=[
  _descriptor.MethodDescriptor(
    name='Ping',
    full_name='FlightControl.Ping',
    index=0,
    containing_service=None,
    input_type=_EMPTY,
    output_type=_EMPTY,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='Shutdown',
    full_name='FlightControl.Shutdown',
    index=1,
    containing_service=None,
    input_type=_EMPTY,
    output_type=_EMPTY,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_FLIGHTCONTROL)

DESCRIPTOR.services_by_name['FlightControl'] = _FLIGHTCONTROL

# @@protoc_insertion_point(module_scope)
