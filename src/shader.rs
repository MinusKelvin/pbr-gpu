use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::Result;

pub fn load_shader(
    device: &wgpu::Device,
    path: &str,
    flags: &HashMap<String, String>,
) -> Result<wgpu::ShaderModule> {
    let mut output = String::new();

    read_shader(&mut output, path.as_ref(), flags, &mut HashSet::new())?;

    Ok(device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(path),
        source: wgpu::ShaderSource::Wgsl(output.into()),
    }))
}

fn read_shader(
    output: &mut String,
    name: &Path,
    flags: &HashMap<String, String>,
    already_included: &mut HashSet<PathBuf>,
) -> Result<()> {
    let path = Path::new("shaders").join(name);
    if already_included.contains(&path) {
        return Ok(());
    }
    let text = std::fs::read_to_string(&path)
        .map_err(|e| std::io::Error::other(format!("{e} `{}`", path.display())))?;
    already_included.insert(path);

    pre_process(
        output,
        text.lines().enumerate(),
        &name,
        flags,
        already_included,
    )
}

fn pre_process<'a>(
    output: &mut String,
    mut lines: impl Iterator<Item = (usize, &'a str)>,
    in_file: &Path,
    flags: &HashMap<String, String>,
    already_included: &mut HashSet<PathBuf>,
) -> Result<()> {
    while let Some((i, line)) = lines.next() {
        if !line.starts_with("#") {
            output.push_str(line);
            output.push('\n');
            continue;
        }

        let mut words = line.split_whitespace();

        match words.next().unwrap() {
            "#import" => {
                let path = words
                    .next()
                    .ok_or_else(|| error(in_file, i, "expected path to import"))?;
                let path = resolve_path(in_file, i, path)?;
                read_shader(output, &path, flags, already_included)?;
            }

            "#importif" => {
                let key = words
                    .next()
                    .ok_or_else(|| error(in_file, i, "expected key to check"))?;
                let value = words
                    .next()
                    .ok_or_else(|| error(in_file, i, "expected value to compare"))?;
                let path = words
                    .next()
                    .ok_or_else(|| error(in_file, i, "expected path to import"))?;

                if flags.get(key).map(String::as_str) == Some(value) {
                    let path = resolve_path(in_file, i, path)?;
                    read_shader(output, &path, flags, already_included)?;
                }
            }

            _ => anyhow::bail!("{}:{}: unrecognized directive", in_file.display(), i + 1),
        }
    }

    Ok(())
}

fn resolve_path(in_file: &Path, i: usize, path: &str) -> Result<PathBuf> {
    let mut new_path = in_file.parent().unwrap().to_path_buf();
    for component in Path::new(&path).components() {
        match component {
            std::path::Component::Prefix(_) => {
                return Err(error(in_file, i, "path prefixes are invalid"));
            }
            std::path::Component::RootDir => new_path.clear(),
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if !new_path.pop() {
                    return Err(error(in_file, i, "navigating to parent of root is invalid"));
                }
            }
            std::path::Component::Normal(os_str) => new_path.push(os_str),
        }
    }
    Ok(new_path)
}

fn error(path: &Path, i: usize, reason: &str) -> anyhow::Error {
    anyhow::anyhow!(format!("{}:{}: {reason}", path.display(), i + 1))
}
