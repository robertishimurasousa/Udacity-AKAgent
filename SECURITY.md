# Security policy

The current `main` branch is maintained for this Udacity coursework project. UDA-Hub is a local demonstration that uses synthetic CultPass data.

## Report a vulnerability

Use [GitHub private vulnerability reporting](https://github.com/robertishimurasousa/Udacity-AKAgent/security/advisories/new) to send a report to the repository maintainer. Include the affected commit, the behavior and its impact, and steps that reproduce it with synthetic data. Keep reports private while the maintainer reviews them.

Use redacted logs or synthetic fixtures when sharing evidence. Keep API keys, environment files, real customer records, and database dumps out of reports and public issues. If an API key was exposed, revoke it through the provider dashboard.

## Local development

Keep provider credentials in the ignored `.env` file or a local secret store. Run the notebooks with the included synthetic inputs. Account tool calls require a verified customer context supplied by the caller.
