//! Minimal GGUF v2/v3 container reader over an in-memory byte slice.
//!
//! Only what the model needs: metadata KVs (strings, scalars, arrays) and
//! the tensor directory (name, shape, quant type, file offset). Tensor data
//! is accessed as subslices of the original buffer — no copies, no mmap.

use crate::error::EngineError;

pub const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// Default data-section alignment when `general.alignment` is absent.
pub const DEFAULT_ALIGNMENT: u64 = 32;

pub(crate) struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], EngineError> {
        let end = self.pos.checked_add(n).ok_or(EngineError::Truncated("size overflow"))?;
        if end > self.bytes.len() {
            return Err(EngineError::Truncated("ran past end of file"));
        }
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32, EngineError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64, EngineError> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn i32(&mut self) -> Result<i32, EngineError> {
        Ok(self.u32()? as i32)
    }

    fn string(&mut self) -> Result<String, EngineError> {
        let n = self.u64()? as usize;
        let bytes = self.take(n)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| EngineError::Truncated("bad utf-8"))
    }
}

/// Metadata value (scalars carry the raw bits; arrays are typed vectors).
#[derive(Debug, Clone)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(MetaArray),
    U64(u64),
    I64(i64),
    F64(f64),
}

#[derive(Debug, Clone)]
pub enum MetaArray {
    Strings(Vec<String>),
    I32(Vec<i32>),
    U32(Vec<u32>),
    F32(Vec<f32>),
    /// Array types the engine reads but does not interpret.
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorInfo {
    /// Quantized element type id (GGUF `ggml_type`).
    pub quant: u32,
}

/// One tensor's directory entry.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    /// Row-major dimensions, GGUF order: dims[0] is the contiguous (row)
    /// length, dims[1] the number of rows.
    pub dims: Vec<u64>,
    pub quant: u32,
    /// Byte offset of the tensor data **within the data section**.
    pub data_offset: u64,
}

/// Parsed GGUF file.
pub struct GgufFile<'a> {
    pub version: u32,
    pub metadata: Vec<(String, MetaValue)>,
    pub tensors: Vec<Tensor>,
    bytes: &'a [u8],
    data_start: u64,
    alignment: u64,
}

impl<'a> GgufFile<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, EngineError> {
        let mut r = Reader::new(bytes);
        if r.take(4)? != GGUF_MAGIC {
            return Err(EngineError::BadMagic);
        }
        let version = r.u32()?;
        if !(2..=3).contains(&version) {
            return Err(EngineError::UnsupportedVersion(version));
        }
        let tensor_count = r.u64()? as usize;
        let kv_count = r.u64()? as usize;

        let mut metadata = Vec::with_capacity(kv_count);
        for _ in 0..kv_count {
            let key = r.string()?;
            let value_type = r.u32()?;
            let value = read_value(&mut r, value_type)?;
            metadata.push((key, value));
        }

        let alignment = match metadata
            .iter()
            .find(|(k, _)| k == "general.alignment")
            .map(|(_, v)| v)
        {
            Some(MetaValue::U32(a)) if *a > 0 => u64::from(*a),
            Some(MetaValue::U64(a)) if *a > 0 => *a,
            _ => DEFAULT_ALIGNMENT,
        };

        let mut tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = r.string()?;
            let n_dims = r.u32()? as usize;
            if n_dims == 0 || n_dims > 4 {
                return Err(EngineError::Truncated("bad tensor rank"));
            }
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(r.u64()?);
            }
            let quant = r.u32()?;
            let data_offset = r.u64()?;
            tensors.push(Tensor {
                name,
                dims,
                quant,
                data_offset,
            });
        }

        // Data section starts at the next alignment boundary.
        let info_end = r.pos as u64;
        let data_start = info_end.div_ceil(alignment) * alignment;

        Ok(Self {
            version,
            metadata,
            tensors,
            bytes,
            data_start,
            alignment,
        })
    }

    pub fn meta(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn meta_str(&self, key: &str) -> Option<&str> {
        match self.meta(key) {
            Some(MetaValue::String(s)) => Some(s),
            _ => None,
        }
    }

    pub fn meta_u32(&self, key: &str) -> Option<u32> {
        match self.meta(key) {
            Some(MetaValue::U32(v)) => Some(*v),
            Some(MetaValue::I32(v)) if *v >= 0 => Some(*v as u32),
            Some(MetaValue::U64(v)) if *v <= u32::MAX as u64 => Some(*v as u32),
            _ => None,
        }
    }

    pub fn meta_f32(&self, key: &str) -> Option<f32> {
        match self.meta(key) {
            Some(MetaValue::F32(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn meta_strings(&self, key: &str) -> Option<&[String]> {
        match self.meta(key) {
            Some(MetaValue::Array(MetaArray::Strings(v))) => Some(v),
            _ => None,
        }
    }

    pub fn meta_i32s(&self, key: &str) -> Option<&[i32]> {
        match self.meta(key) {
            Some(MetaValue::Array(MetaArray::I32(v))) => Some(v),
            _ => None,
        }
    }

    pub fn architecture(&self) -> Option<&str> {
        self.meta_str("general.architecture")
    }

    /// The tensor's bytes from the data section (must be a parsed tensor).
    pub fn tensor_data(&self, tensor: &Tensor) -> Result<&'a [u8], EngineError> {
        let start = self.data_start + tensor.data_offset;
        let len = tensor_byte_len(tensor, self.alignment)?;
        let end = start.checked_add(len).ok_or(EngineError::Truncated("tensor overflow"))?;
        if end as usize > self.bytes.len() {
            return Err(EngineError::Truncated("tensor data past end of file"));
        }
        Ok(&self.bytes[start as usize..end as usize])
    }
}

/// Number of bytes the tensor's data occupies in the file (padded to the
/// section alignment only between tensors — each tensor's own extent is
/// `type_size`-padded per row, which for our types divides evenly).
pub fn tensor_byte_len(tensor: &Tensor, _alignment: u64) -> Result<u64, EngineError> {
    let n_elems: u64 = tensor.dims.iter().product();
    let row_len = *tensor.dims.first().ok_or(EngineError::Truncated("empty tensor"))?;
    let n_rows: u64 = tensor.dims.iter().skip(1).product();
    let row_len = if row_len == 0 { 1 } else { row_len };

    // Per-type: bytes per block and elements per block (block = row chunk).
    let (block_elems, block_bytes) = quant_block(tensor.quant)
        .ok_or_else(|| EngineError::UnsupportedQuant {
            tensor: tensor.name.clone(),
            quant: tensor.quant,
        })?;

    let blocks_per_row = row_len.div_ceil(block_elems);
    // All supported quants require whole blocks per row (GGUF guarantee for
    // llama-family tensors we load).
    if row_len % block_elems != 0 {
        return Err(EngineError::BadTensorShape {
            tensor: tensor.name.clone(),
        });
    }
    let _ = n_elems;
    Ok(blocks_per_row * block_bytes * n_rows.max(1))
}

/// (elements per block, bytes per block) for the quant types the engine
/// supports loading. Matches llama.cpp's block layouts; anything else is
/// rejected at load with `UnsupportedQuant`.
pub fn quant_block(quant: u32) -> Option<(u64, u64)> {
    match quant {
        0 => Some((1, 4)),           // F32
        1 => Some((1, 2)),           // F16
        2 => Some((32, 18)),         // Q4_0: f16 scale + 16 nibble bytes
        8 => Some((32, 34)),         // Q8_0: f16 scale + 32 i8
        14 | 15 => Some((256, 144)), // Q4_K_S / Q4_K_M
        18 => Some((256, 210)),      // Q6_K
        _ => None,
    }
}

fn read_value(r: &mut Reader<'_>, value_type: u32) -> Result<MetaValue, EngineError> {
    let value = match value_type {
        0 => MetaValue::U8(r.take(1)?[0]),
        1 => MetaValue::I8(r.take(1)?[0] as i8),
        2 => {
            let b = r.take(2)?;
            MetaValue::U16(u16::from_le_bytes([b[0], b[1]]))
        }
        3 => {
            let b = r.take(2)?;
            MetaValue::I16(i16::from_le_bytes([b[0], b[1]]))
        }
        4 => MetaValue::U32(r.u32()?),
        5 => MetaValue::I32(r.i32()?),
        6 => {
            let b = r.take(4)?;
            MetaValue::F32(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        }
        7 => MetaValue::Bool(r.take(1)?[0] != 0),
        8 => MetaValue::String(r.string()?),
        9 => {
            let elem_type = r.u32()?;
            let n = r.u64()? as usize;
            let array = match elem_type {
                8 => MetaArray::Strings((0..n).map(|_| r.string()).collect::<Result<Vec<_>, _>>()?),
                5 => {
                    let mut v = Vec::with_capacity(n);
                    for _ in 0..n {
                        v.push(r.i32()?);
                    }
                    MetaArray::I32(v)
                }
                4 => {
                    let mut v = Vec::with_capacity(n);
                    for _ in 0..n {
                        v.push(r.u32()?);
                    }
                    MetaArray::U32(v)
                }
                6 => {
                    let mut v = Vec::with_capacity(n);
                    for _ in 0..n {
                        let b = r.take(4)?;
                        v.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
                    }
                    MetaArray::F32(v)
                }
                _ => {
                    // Skip unknown element types by their fixed size.
                    let size = match elem_type {
                        0 | 1 | 7 => 1,
                        2 | 3 => 2,
                        4 | 5 | 6 => 4,
                        10 | 11 | 12 => 8,
                        _ => return Err(EngineError::Truncated("bad meta array type")),
                    };
                    for _ in 0..n {
                        r.take(size)?;
                    }
                    MetaArray::Other
                }
            };
            MetaValue::Array(array)
        }
        10 => {
            let b = r.take(8)?;
            MetaValue::U64(u64::from_le_bytes([
                b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
            ]))
        }
        11 => {
            let b = r.take(8)?;
            MetaValue::I64(i64::from_le_bytes([
                b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
            ]))
        }
        12 => {
            let b = r.take(8)?;
            MetaValue::F64(f64::from_le_bytes([
                b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
            ]))
        }
        _ => return Err(EngineError::Truncated("bad meta value type")),
    };
    Ok(value)
}
