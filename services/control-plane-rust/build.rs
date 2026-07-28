fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_root = "../../proto";
    tonic_build::configure().compile(
        &[
            format!("{}/polymas/v1/patient.proto", proto_root),
            format!("{}/polymas/v1/services.proto", proto_root),
        ],
        &[proto_root],
    )?;
    Ok(())
}
