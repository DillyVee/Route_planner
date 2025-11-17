# Security Policy

## Supported Versions

We release security updates for the following versions of Route Planner:

| Version | Supported          |
| ------- | ------------------ |
| 4.0.x   | :white_check_mark: |
| 3.x.x   | :x:                |
| < 3.0   | :x:                |

## Reporting a Vulnerability

We take the security of Route Planner seriously. If you believe you have found a security vulnerability, please report it to us responsibly.

### How to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via one of the following methods:

1. **GitHub Security Advisories** (Preferred)
   - Go to https://github.com/DillyVee/Route_planner/security/advisories/new
   - Fill out the advisory form with details about the vulnerability

2. **Email**
   - Send details to: [INSERT SECURITY EMAIL]
   - Use subject line: "[SECURITY] Route Planner Vulnerability Report"

### What to Include

Please include as much of the following information as possible:

- **Type of vulnerability** (e.g., SQL injection, code execution, DoS)
- **Component affected** (e.g., KML parser, routing algorithm, visualization)
- **Steps to reproduce** the vulnerability
- **Proof of concept** code or exploit (if available)
- **Impact assessment** - How severe is this vulnerability?
- **Suggested remediation** (if you have ideas)

### Response Timeline

- **Initial Response**: Within 48 hours of report submission
- **Status Update**: Within 7 days with assessment and timeline
- **Patch Development**: Varies based on severity and complexity
- **Public Disclosure**: After patch is released and users have time to update

### Security Update Process

1. **Vulnerability Confirmed**: Maintainers verify and assess severity
2. **Patch Developed**: Fix is developed in private repository
3. **Testing**: Patch is tested with comprehensive test suite
4. **Advisory Published**: GitHub Security Advisory is created
5. **Patch Released**: New version published with security fixes
6. **Public Disclosure**: Details published 7-14 days after release

## Security Best Practices for Users

### Input Validation

- **KML Files**: Always validate KML files from untrusted sources
- **Sanitization**: The pipeline includes XML sanitization, but be cautious with malformed files
- **File Size**: Large KML files (>100MB) should be pre-validated

### Dependencies

- **Keep Updated**: Regularly update Route Planner and all dependencies
- **Check for CVEs**: Monitor security advisories for dependencies

```bash
# Check for known vulnerabilities
pip install safety
safety check -r requirements_production.txt
```

### Execution Environment

- **Sandboxing**: Consider running route optimization in isolated environment for untrusted KML
- **Resource Limits**: Set memory and CPU limits for large dataset processing
- **File Permissions**: Ensure output directories have appropriate permissions

### Data Privacy

- **Sensitive Routes**: Be aware that visualization files may contain sensitive location data
- **KML Metadata**: MapPlus files may contain proprietary roadway data
- **Output Sharing**: Review generated files before sharing publicly

## Known Security Considerations

### XML External Entity (XXE) Attacks

- **Status**: Mitigated
- **Details**: KML parser uses `xml.etree.ElementTree` with default safe settings
- **Recommendation**: Do not modify XML parser settings to allow external entities

### Denial of Service (DoS)

- **Large Datasets**: Processing 10,000+ segments requires significant CPU/memory
- **Mitigation**: Use clustering and on-demand routing mode
- **Recommendation**: Implement timeouts and resource limits for untrusted inputs

### Code Injection

- **Status**: Low risk
- **Details**: No dynamic code evaluation of user inputs
- **Recommendation**: Do not use `eval()` or `exec()` on any user-provided data

### Path Traversal

- **Status**: Mitigated
- **Details**: File operations use `pathlib.Path` with validation
- **Recommendation**: Validate all user-provided file paths

## Disclosure Policy

We follow **Coordinated Vulnerability Disclosure**:

- Security issues are fixed in private before public disclosure
- Reporters are credited (if desired) in security advisories
- CVE IDs are requested for significant vulnerabilities
- Public disclosure occurs after patches are available

## Security Hall of Fame

We recognize security researchers who responsibly disclose vulnerabilities:

- *Your name could be here!*

## Security Updates

Subscribe to security notifications:
- Watch this repository for security advisories
- Enable GitHub security alerts
- Follow releases for security patches

## Contact

For security concerns, contact:
- GitHub Security Advisories: https://github.com/DillyVee/Route_planner/security/advisories
- Email: [INSERT SECURITY EMAIL]

---

**Last Updated**: 2025-11-17
