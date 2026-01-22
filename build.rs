use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    lalrpop::Configuration::new()
        .emit_rerun_directives(true)
        .set_in_dir("src")
        .process()
}
