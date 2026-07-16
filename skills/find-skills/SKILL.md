---
name: find-skills
description: Use when the developer asks whether a reusable Agent Skill, Copilot customization, plugin, extension, MCP server, or workflow exists for a particular capability.
argument-hint: "[capability or workflow]"
---

# Find Skills

Find or propose reusable Copilot capabilities for the requested workflow.

## Process

1. Identify the exact capability the developer needs.
2. Determine whether it should be implemented as:
   - an Agent Skill,
   - a custom agent,
   - workspace instructions,
   - an MCP server,
   - a VS Code extension,
   - an Agent Plugin.
3. Prefer simple workspace Agent Skills when Markdown instructions are enough.
4. Prefer established and actively maintained sources.
5. Review the complete skill and all referenced files before recommending it.
6. Check for:
   - shell commands,
   - scripts,
   - package installation,
   - network access,
   - hooks,
   - MCP servers,
   - access to credentials,
   - access to environment variables.
7. Explain whether it works in:
   - lightweight browser VS Code,
   - GitHub Codespaces,
   - desktop VS Code.
8. Do not install or execute third-party code automatically.
9. For workspace installation, place the complete skill directory under:

   `.github/skills/<skill-name>/`

10. Ensure the directory name matches the `name` field in `SKILL.md`.

## Output

Provide:

- Recommended skill or approach
- Why it matches the request
- Browser compatibility
- Security considerations
- Required installation steps
- Any limitations
