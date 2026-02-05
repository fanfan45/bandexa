# Privacy

This document describes Bandexa’s privacy posture and the privacy implications of different deployment modes.

## Summary

- **Local by default.** By default, Bandexa runs locally and does not collect telemetry or “phone home.”
- **Cloud is user-configured.** If you configure cloud storage or deploy components across machines (e.g., Kubernetes + S3), your data may leave the local environment as a direct result of your configuration and infrastructure choices.
- **You own deployment privacy and security.** In any non-local deployment, you are responsible for privacy, security, and compliance obligations related to your data and infrastructure.

## What Bandexa does and does not do

### Bandexa does not:
- collect telemetry
- transmit data to the project maintainers
- include built-in analytics or usage tracking
- automatically upload your data anywhere

### Bandexa does:
- process the data **you provide** to it (contexts, actions, rewards, etc.) in order to run contextual bandit policies and update models/posteriors
- write data to storage only when **you configure** a storage backend that does so (e.g., files, object stores, databases) or when your deployment environment persists data

## Local mode vs cloud-enabled mode

### Local mode
In local mode, data and artifacts remain on your machine *unless you explicitly export, upload, or integrate Bandexa with remote services.*

### Cloud-enabled mode (user-configured)
“Cloud support” means Bandexa can be used in deployments where data and artifacts are stored or exchanged via remote systems (for example, object storage such as S3, remote filesystems, or networked services), and where components may run on different machines or containers (for example, action selection and training running in separate Kubernetes pods).

In these deployments, **data may be written to or read from remote systems** as part of your architecture.

## What cloud support means for privacy (what data may leave local)

If you configure remote storage (e.g., S3 buckets) or run Bandexa across machines, the following categories of data may be written to or read from those systems, depending on how you integrate Bandexa:

- **Contexts** (raw context vectors and/or serialized context payloads)
- **Action features and/or action IDs** (raw action vectors, action embeddings, identifiers, metadata)
- **Rewards** (observed rewards, logged outcomes, labels)
- **Logs** (application logs, debug logs, traces produced by your runtime or orchestration)
- **Model artifacts** (encoder weights, posterior parameters, policy state)
- **Checkpoints** (snapshots of training state)
- **Embeddings / features** (learned representations derived from contexts/actions)

Bandexa does not decide whether these items are “sensitive.” That depends on your use case and what you encode into contexts/actions, logs, and artifacts.

## Privacy and security risks introduced by cloud or distributed deployments

If contexts/actions can contain **PII or sensitive information**, then storing or transporting them via cloud systems or across networks introduces dangerous additional privacy and security risk surfaces. Examples include, but are not limited to:

- **Misconfiguration of remote storage** (e.g., accidental public exposure or overly permissive access)
- **Credential compromise** (e.g., leaked keys/tokens or unintended credential distribution)
- **Overly broad authorization policies** (e.g., identities that can access more data than intended)
- **Logs capturing sensitive data** (e.g., request/response logging, debugging output, stack traces)
- **Network exposure** (e.g., unintended data transmission across networks or between services)
- **Retention and persistence surprises** (e.g., backups, replicas, snapshots, versioning, or delayed deletion)
- **Multi-tenant infrastructure risks** (e.g., shared environments and access boundary mistakes)

These risks arise from your deployment environment and configuration. Bandexa **does not manage** these controls for you.

## Responsibility boundary (important)

Bandexa is a local library and does not collect telemetry or transmit data to the project maintainers. If you configure Bandexa to use remote storage (e.g., S3 buckets) or deploy components across machines (e.g., Kubernetes pods for action selection and training), data such as contexts, action features/IDs, rewards, logs, and model artifacts may be written to or read from those systems. In such deployments, **you are responsible** for authentication/authorization controls, encryption choices, bucket/service policies, retention/deletion practices, access logging, and compliance with any applicable privacy or security requirements.

Bandexa does not provide compliance guarantees or security configuration for your cloud environment. You are responsible for securing credentials, access policies, network boundaries, and for evaluating whether your data is appropriate to store in remote services.

## Your responsibilities (non-exhaustive)

If you use Bandexa in any environment where data may leave the local machine (including cloud storage, distributed services, or shared infrastructure), you are responsible for (including but not limited to):

- obtaining consents / lawful basis for processing (if applicable)
- ensuring compliance with applicable privacy and security requirements (e.g., HIPAA/PII/PHI/GDPR or other regimes relevant to your domain)
- configuring and maintaining access controls (including least-privilege principles where applicable to your system)
- choosing and configuring encryption in transit (e.g., TLS) and at rest (e.g., server-side encryption such as SSE-S3 or SSE-KMS where relevant)
- preventing unintended public exposure of stored data or artifacts
- defining and enforcing retention/deletion policies appropriate for your data and obligations
- auditing and monitoring access logs and data access patterns as needed for your environment

## Changes to this document

This document may evolve as Bandexa adds optional integrations or storage backends. The guiding principle remains:

**Bandexa is local by default, does not phone home, and any cloud or distributed behavior is user-configured and user-owned.**
