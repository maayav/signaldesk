# Security guidance

SignalDesk is a local ML engineering demonstration. The current API has input/body limits and validates every manifest-listed artifact checksum before loading the model, but it does not include authentication, user isolation, or a managed production perimeter. Local launch and Compose defaults bind host ports to the loopback interface.

## Deployment boundaries

- Load only model artifacts from a source you trust. Joblib deserialization can execute code. A checksum detects corruption; replacing a model and its manifest together can bypass that integrity check.
- Keep the API private unless the deployment adds suitable authentication, HTTPS, request limits, and operational controls.
- The event store excludes submitted ticket text. Requests still pass through process memory; reverse-proxy and platform logging need their own configuration.
- Do not use real customer secrets in demo inputs or issue reports. Keep event databases, `.env` files, and credentials out of version control.

See the [architecture](docs/architecture.md) for the implemented behavior and the [model card](docs/model-card.md) for prediction limitations. Confidence and drift indicators are not security controls.

## Report a potential vulnerability

Check the repository’s [Security tab](https://github.com/maayav/signaldesk/security). If private vulnerability reporting is enabled, use it for sensitive technical details. Otherwise, open a minimal issue requesting a private reporting channel; do not publish exploit details, credentials, private ticket text, or personal data in that issue.

Include the affected version or commit, a minimal reproduction with synthetic inputs, the expected impact, and any mitigation you tested. This project does not promise a response or patch timeline.
