use serde::{Deserialize, Serialize};
use serde_json;
use std::path::{PathBuf};
use std::env;
use std::fs;
use walkdir::{DirEntry, WalkDir};
use num_traits::{Float, FloatConst, NumAssignOps};

#[allow(dead_code)]

struct PathManage {
    _root: PathBuf
}

impl PathManage {

    fn new(root: &str) -> PathManage {
        PathManage{_root: PathManage::find_root(root).unwrap()}
    }

    fn split(path: &PathBuf) -> Vec<&str> {
        path.components().map(|c| c.as_os_str().to_str().unwrap()).collect::<Vec<&str>>()
    }

    fn is_hidden(entry: &DirEntry) -> bool {
        entry.file_name()
             .to_str()
             .map(|s| s.starts_with("."))
             .unwrap_or(false)
    }

    fn path(&self, path: &str) -> Option<PathBuf> {
        let path_pathbuf = PathBuf::from(path);
        let path_parts = Self::split(&path_pathbuf);
        let path_parts: Vec<&str> = path_parts.into_iter().filter(|&part| part != "." && part != ".." && !part.is_empty()).collect();
        let length = path_parts.len();

        for entry in WalkDir::new(self._root
                                                    .as_os_str()
                                                    .to_str().unwrap())
                                                    .into_iter()
                                                    .filter_map(|e| e.ok()) 
        {
            let entry_path: PathBuf = entry.path().to_path_buf();
            let entry_parts = Self::split(&entry_path);
            if (entry_parts.len() as i32) - (length as i32) < 1 { // Avoid neg index and cast to i32 to avoid overflows
                continue;
            }
            let semipath = entry_parts[entry_parts.len() - length ..].to_vec();
            if path_parts.iter().all(|item| semipath.contains(item)) {
                return Some(path_pathbuf);
            }

        }

        None
    }

    fn find_root(root_name: &str) -> Result<PathBuf, std::io::Error> {
        let current_dir = env::current_dir()?;
        let mut path = current_dir.components().collect::<Vec<_>>();
        path.reverse();

        let mut dir = PathBuf::new();
        dir.push(".");

        for component in path {
            if let Some(name) = component.as_os_str().to_str() {
                if name != root_name {
                    dir.push("..");
                } else {
                    return Ok(dir);
                }
            }
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "Root directory not found",
        ))
    }
}

#[derive(Deserialize, Serialize, Debug)]
pub enum Data<T> {
    FftFreqVals {
        n: T,
        d: T
    },

    ComplexVals {
        mag: Vec<T>,
        phase: Vec<T>
    },

    Array(Vec<T>),

    SineFreqVals {
        fsine: T,
        fsample: T,
        duration: T
    }
}

#[derive(Deserialize, Serialize, Debug)]
pub struct Json<F: Float + FloatConst + NumAssignOps + 'static> {
    pub input_data: Data<F>,
    pub output_data: Data<F>, 
    pub function: String,
    pub path: String
}

pub fn read_json<'a, F: Float + FloatConst + NumAssignOps + 'static + for<'de> Deserialize<'de>>(lib_path: &str) -> Json<F> {
    let path = PathManage::new("rufft");
    let json_path = path.path(lib_path).unwrap();
    let file = fs::File::open(json_path).unwrap();
    let data: Json<F> = serde_json::from_reader(file).unwrap();
    return data;
}
