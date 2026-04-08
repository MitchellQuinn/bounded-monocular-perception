
## Pack structure

```text
codex_templates/
├── README.md
├── 00-template-governance.md
├── 01-base-task-template.md
├── 02-ui-consistency-contract.md
├── 03-feature-addition-template.md
├── 04-repair-refactor-template.md
├── 05-review-template.md
├── 06-ui-vocabulary.md
├── 07-notebook-widget-style-template.md
├── 08-training-session-control-template.md
├── 09-result-object-contract.md
└── 10-command-construction-contract.md
```

---

## `README.md`

```md
# Codex Template Pack Index

## Purpose
This file explains which template(s) to use for which kind of Codex task.

## Template Selector

### Use `01-base-task-template.md`
When:
- starting a new task from scratch
- defining a fresh feature, integration, or refactor contract
- you need the general contract without task-specific extras

### Use `03-feature-addition-template.md`
When:
- extending an existing tool
- adding controls or behaviour to an existing notebook control panel
- preserving an existing structure matters

### Use `04-repair-refactor-template.md`
When:
- fixing structure drift
- separating concerns that got muddled
- moving leaked UI layout back into the notebook
- removing scope creep or helper/UI boundary violations

### Use `05-review-template.md`
When:
- auditing an implementation without rewriting it yet
- checking architectural boundaries
- checking UI consistency
- checking notebook/helper leakage
- deciding what to fix first

## Required Pairings

### For any notebook UI task, always pair with:
- `02-ui-consistency-contract.md`
- `06-ui-vocabulary.md`
- `07-notebook-widget-style-template.md`

### For process-launch / session-control tasks, also pair with:
- `09-result-object-contract.md`
- `10-command-construction-contract.md`

### For training/session-control notebook tasks, prefer:
- `08-training-session-control-template.md`

## Current House Rules
- notebook owns UI layout
- helper owns logic and system interaction
- process-launch behaviour must not be implemented directly in notebook UI code
- helper results should use the standard result object contract where applicable
- command construction should follow the command construction contract
- destructive actions require explicit selection and explicit confirmation
- refresh behaviour must be explicit and must preserve editable input state unless explicitly stated otherwise

## Output Rule
Default preference:
- return complete contents of each new or materially changed file

Escape hatch:
- for very large files that are mostly unchanged, return complete contents only for new or materially changed files unless explicitly asked to regenerate the entire file

## Recommended Usage Pattern
1. choose the main task template
2. add the required paired contracts
3. add task-specific notes
4. keep scope narrow
5. review with `05-review-template.md` after implementation if the task is important
```

---

## `00-template-governance.md`

```md
# Codex Template Governance

## Purpose
This template pack exists to improve consistency across Codex-generated implementations, especially for notebook-based control panels and lightweight operational tools.

The goals are:
- preserve architectural boundaries
- improve UI consistency across iterations
- reduce one-off interface improvisation
- keep prompts versionable and auditable
- keep notebook UI layout easy to tweak manually
- prevent UI structure from leaking into helper modules
- keep helper returns consistent and easy to wire into notebook event handlers

## Core Principles
- Prefer stable patterns over novelty.
- Prefer explicit contracts over implied intent.
- Prefer complete-file outputs over patch fragments.
- Prefer deterministic naming and traceable behaviour.
- Prefer narrow version-1 scope over speculative expansion.
- Prefer reporting template gaps over inventing one-off solutions.

## Non-Negotiable UI Boundary
For notebook-based tools:

- The **Jupyter notebook owns UI layout**.
- The **Python helper layer owns logic and system interaction**.

### The notebook layer is responsible for:
- widget creation
- widget arrangement
- panel ordering
- labels and visual grouping
- button placement
- table/list placement
- status-area placement
- event wiring between notebook widgets and helper calls

### The helper layer is responsible for:
- validation
- business logic
- command construction
- system interaction
- structured results
- traceable errors

### The helper layer must not:
- define notebook layout
- build notebook panels
- return preassembled notebook widget trees
- control UI grouping or visual structure
- become a presentation layer

## Standard Result Shape
Where applicable, helper functions should return the standard result object contract defined in:
- `09-result-object-contract.md`

This keeps notebook event handlers simpler and more consistent.

## Command Construction
For process-launch tasks, command building must follow:
- `10-command-construction-contract.md`

Command construction is its own concern.
It must not be mixed into notebook layout code.

## Standard Output Contract
Unless explicitly overridden, Codex must return:
1. a short implementation summary
2. assumptions made
3. complete contents of each new or changed file
4. deferred TODOs or risks
5. any suggested template improvements if the task exposed inconsistency

### Escape Hatch
For very large files that are mostly unchanged:
- prefer complete contents only for new or materially changed files unless explicitly asked to regenerate the entire file

## Standard Implementation Rules
- Do not invent new UI patterns unless explicitly requested.
- Do not silently broaden the scope.
- Do not collapse distinct responsibilities into one function or module.
- Do not put process-management logic into the notebook UI layer.
- Do not put notebook layout into the Python helper layer.
- Do not rely on print spam as a substitute for interface design.
- Do not return patch fragments unless explicitly requested.

## Destructive Action Rule
Destructive actions require:
- explicit item selection
- explicit confirmation
- clear status reporting

Do not attach destructive behaviour to implicit refresh or ambiguous action buttons.

## Refresh Rule
Notebook tools must define:
- what refresh updates
- whether refresh is manual or automatic
- whether refresh preserves editable input state

Default rule:
- refresh is explicit
- refresh preserves editable input state

## Template Versioning
Every task prompt should include:
- template name
- template version
- task type
- target area

When the house style changes:
- increment the relevant template version
- keep old versions available until deliberately retired
- document why the change was made

## UI House Style
Unless explicitly overridden, notebook control-panel tasks should use this layout order:

1. Header / context area
2. Actions panel
3. Status panel
4. Results / list / table panel

This ordering should remain stable across tasks.

## Required Improvement Behaviour
If Codex encounters friction caused by ambiguity or inconsistency in the template:
- call it out explicitly
- suggest a concrete template improvement
- do not solve it by inventing a one-off pattern

## Scope Discipline
For early versions:
- implement only what is requested
- do not add future-phase features
- do not add speculative abstractions unless they materially simplify the contract
```

---

## `01-base-task-template.md`

```md
# Codex Base Task Template

## Template Metadata
- Template name: [name]
- Template version: [version]
- Task type: [feature / refactor / repair / review / UI polish / integration]
- Target area: [module / notebook / control panel / helper]

## Objective
Implement [feature/component] for [project/system name].

## Problem Statement
We need [clear description of what should exist when this task is done].

## Scope
In scope:
- [item]
- [item]
- [item]

Out of scope:
- [item]
- [item]
- [item]

## Architectural Boundaries
Preserve the following separation of concerns:

### Notebook UI layer
Responsibilities:
- widget creation
- layout and visual grouping
- panel ordering
- action-button placement
- rendering status and results
- wiring widget events to helper calls

Must not:
- contain core process logic
- become the main system-logic layer

### Helper / logic layer
Responsibilities:
- validation
- command construction where relevant
- system interaction
- structured return values
- traceable errors

Must not:
- define notebook layout
- assemble notebook widget panels
- return prebuilt notebook UI unless explicitly requested

### Execution / persistence / external layer
Responsibilities:
- external execution
- persistence
- runtime/system state

Must not:
- act as UI
- contain notebook presentation logic

Do not conflate these responsibilities.

## Functional Contract
Inputs:
- [input]
- [input]

Outputs:
- [output]
- [output]

Required behaviour:
- [behaviour]
- [behaviour]
- [behaviour]

Must not:
- [behaviour to avoid]
- [behaviour to avoid]

## UI Contract
If this task includes user-facing notebook UI, it must follow:
- `02-ui-consistency-contract.md`
- `06-ui-vocabulary.md`
- `07-notebook-widget-style-template.md`

Important rule:
- notebook owns UI layout
- helper layer owns logic
- do not move UI composition into helper modules

If this task does not include UI, say so explicitly and do not invent any.

## Result Object Contract
Where helper functions return structured status, use:
- `09-result-object-contract.md`

## Command Construction Contract
For process-launch or command-preview tasks, use:
- `10-command-construction-contract.md`

## Data / Naming Contract
Use [identifier] as the primary identifier throughout:
- [place]
- [place]
- [place]

Naming rules:
- [rule]
- [rule]

## Logging / Traceability
- [logging requirement]
- [logging requirement]
- [traceability requirement]

## Failure Policy
Use fail-fast behaviour where appropriate:
- if [failure], then [expected behaviour]
- if [failure], then [expected behaviour]

Do not silently continue after partial failure.

## Refresh Behaviour
Define explicitly:
- what refresh updates
- whether refresh is manual or automatic
- whether refresh preserves editable user input

Default expectation:
- refresh is explicit
- refresh preserves editable user input

## Deliverables
Produce:
1. [file or component]
2. [file or component]
3. [tests / validation / notes]

## Constraints
- Use [language / framework / tools]
- Do not add unnecessary dependencies
- Keep interfaces crisp
- Keep behaviour deterministic where possible
- Prefer simple, auditable logic over cleverness
- Do not introduce new UI patterns unless explicitly requested
- Do not put notebook layout in Python helpers

## Acceptance Criteria
The task is complete when:
- [criterion]
- [criterion]
- [criterion]

## Output Format
Return:
1. a short implementation summary
2. assumptions made
3. the complete contents of each new or materially changed file
4. deferred TODOs or risks
5. any suggested template improvements if the task exposed ambiguity or inconsistency

Do not return partial patches unless explicitly requested.

For very large files that are mostly unchanged:
- return complete contents only for new or materially changed files unless explicitly asked to regenerate the entire file

## Notes
Additional context:
[freeform notes]
```

---

## `02-ui-consistency-contract.md`

```md
# UI Consistency Contract

## Purpose
This contract defines the standard UI shape for Codex-generated notebook control panels and lightweight operational interfaces.

The goals are:
- preserve interface consistency across tasks
- keep layout editable in the notebook
- prevent visual structure from drifting into helper code

## Hard Boundary
UI layout, widget composition, and panel grouping must remain in the notebook layer.

### The notebook layer should own:
- widget instantiation
- layout containers
- panel ordering
- labels and headings
- action-button placement
- results-table placement
- status-area placement

### The helper layer may provide:
- data structures
- status objects
- validation results
- action functions
- query results

### The helper layer must not:
- assemble notebook layout
- return a fully built widget tree
- decide panel structure
- define visual grouping

## Standard Layout
Unless explicitly told otherwise, use this layout order:

1. **Header**
   - title
   - short description
   - optional current context summary

2. **Actions**
   - editable inputs
   - action buttons
   - derived previews or computed paths if relevant

3. **Status**
   - validation messages
   - action success/failure messages
   - availability / environment errors

4. **Results**
   - selectable list or table
   - refresh action
   - details area only if clearly needed

Do not rearrange these sections unless the template itself is being revised.

## Standard Controls
Prefer the following control types:
- labeled text input
- labeled dropdown/select
- explicit action buttons
- selectable table or list
- dedicated status/message area

Avoid:
- hidden click regions
- print-output pseudo-interfaces
- overloaded controls that do multiple unrelated things
- mixing multiple interaction styles for equivalent actions

## Button Conventions
Use short, literal labels.

Preferred examples:
- `Validate`
- `Launch`
- `Refresh`
- `Stop`
- `Open Log`
- `Run`
- `Save`
- `Apply`
- `Clear`

Avoid vague labels like:
- `Go`
- `Do It`
- `Execute Now`

## Input Conventions
- Inputs must have visible labels.
- Important derived values should be displayed near the related input or action.
- Validation errors should appear near the relevant input and in the status panel if significant.
- Refresh actions must not wipe editable user input unless explicitly requested.

## Status Conventions
Status messages must appear in a predictable, dedicated area.

Categories should remain distinct:
- validation issue
- action success
- action failure
- environment/runtime availability problem

Do not rely on scattered prints for status.

## List / Table Conventions
When showing a list or table:
- keep columns stable across refreshes
- keep ordering predictable
- use one consistent selection mechanism
- keep refresh as an explicit action
- do not silently change representation between iterations

## Destructive Action Rule
Destructive actions require:
- explicit item selection
- explicit confirmation
- clear success/failure reporting in Status

Do not bind destructive actions to refresh or ambiguous controls.

## Refresh Rule
Every notebook UI task must define:
- what refresh updates
- whether refresh is manual or automatic
- whether refresh preserves editable input state

Default rule:
- refresh is manual
- refresh preserves editable input state

## Consistency Rule
If an existing panel, label, or interaction pattern already exists in the current notebook tool, reuse it rather than inventing a new one.

## Improvement Rule
If the current contract causes awkwardness:
- report the friction clearly
- suggest a concrete improvement to the contract
- do not solve it with a one-off UI invention
```

---
# `03-feature-addition-template.md`

```md
# Codex Feature Addition Template

## Template Metadata
- Template name: feature addition
- Template version: v0.4
- Task type: feature
- Target area: [module / notebook / control panel / helper]

## Objective
Add [feature name] to the existing system.

## Existing System Context
The current system already provides:
- [existing capability]
- [existing capability]
- [existing capability]

This task must extend the existing system without breaking its current structure or inventing a parallel pattern.

## Feature Definition
Add support for:
- [feature behaviour]
- [feature behaviour]
- [feature behaviour]

Do not add:
- [non-feature]
- [non-feature]
- [future-phase idea]

## Required Pairings
For notebook UI additions, also pair with:
- `02-ui-consistency-contract.md`
- `06-ui-vocabulary.md`
- `07-notebook-widget-style-template.md`

For process-launch additions, also pair with:
- `09-result-object-contract.md`
- `10-command-construction-contract.md`

## Architectural Boundaries
Preserve the existing separation of concerns:

### Notebook UI layer
Responsibilities:
- widget layout
- action placement
- panel grouping
- status rendering
- result rendering
- user interaction flow

Must not:
- contain core process logic
- absorb helper responsibilities

### Helper / logic layer
Responsibilities:
- validation
- command construction
- system interaction
- structured result generation

Must not:
- define notebook layout
- return prebuilt notebook UI
- embed notebook-specific visual structure

### Execution / persistence layer
Responsibilities:
- execution hosting
- state persistence where relevant
- external system behaviour

Must not:
- act as UI
- become the source of ad hoc business logic

## UI Contract
This feature must reuse the existing notebook control-panel layout:
1. Header / context area
2. Primary action panel
3. Status panel
4. Results / list / table panel

### Required UI behaviour
- add the smallest necessary controls
- preserve existing button order where possible
- preserve existing labels where possible
- keep new controls in the appropriate existing panel
- do not create a new panel unless clearly necessary
- keep layout definition in the notebook, not helper code

## Inputs and Outputs
Inputs:
- [input]
- [input]

Outputs:
- [output]
- [output]

## Data / Naming Contract
Use [identifier] as the primary identifier throughout the new feature.

Rules:
- [rule]
- [rule]

## Failure Policy
- validate before expensive or irreversible actions
- fail clearly on invalid input
- do not silently fall back to different behaviour
- do not partially apply the feature without reporting it

## Deliverables
Produce:
1. updated implementation files
2. any required supporting types/utilities
3. tests or validation notes
4. short usage note if behaviour changed

## Acceptance Criteria
The task is complete when:
- the new feature works
- the existing notebook UI shape remains consistent
- no parallel interaction pattern has been introduced
- layout still lives in the notebook
- the feature remains within declared scope
- notebook layout remains notebook-owned
- helper results remain consistent with the result object contract where applicable
- command construction remains helper-owned where applicable

## Output Format
Return:
1. a short implementation summary
2. assumptions made
3. the complete contents of each new or changed file
4. deferred TODOs or risks
5. any suggested template improvements if the task exposed UI or contract friction

Do not return patch fragments unless explicitly requested.

## Notes
Additional context:
[freeform notes]

```

---

## `04-repair-refactor-template.md`

```md
# Codex Repair / Refactor Template

## Template Metadata
- Template name: repair or refactor
- Template version: v0.4
- Task type: [repair / refactor]
- Target area: [module / notebook / control panel / helper]

## Objective
Repair or refactor the existing implementation so it matches the intended contract.

## Problem Statement
The current implementation has one or more of the following issues:
- [issue]
- [issue]
- [issue]

The goal is to fix the design, not just patch visible symptoms.

## Required Diagnosis
Before regenerating code, identify:
- what is structurally wrong
- which contract boundaries were violated
- whether the UI drifted from the expected pattern
- whether notebook layout leaked into helpers or helpers leaked into notebook logic
- whether the implementation broadened scope improperly
- whether result-object shapes drifted or became inconsistent
- whether command construction leaked into notebook UI code

## Architectural Contract
The repaired version must preserve:

### Notebook UI layer
Responsibilities:
- widget composition
- layout and grouping
- panel ordering
- action placement
- rendering status and results

Must not:
- absorb core process logic
- become the helper layer

### Helper / logic layer
Responsibilities:
- validation
- command construction
- system interaction
- structured result generation

Must not:
- define notebook layout
- return prebuilt notebook UI
- own visual grouping or presentation structure

### Execution / external layer
Responsibilities:
- runtime behaviour
- persistence or external execution

Must not:
- act as UI
- absorb notebook display concerns

## UI Contract
If this task includes notebook UI, repair it to match the UI consistency contract.

That means:
- stable panel layout
- explicit buttons for actions
- predictable status area
- consistent list/table behaviour
- no print-based pseudo-UI
- no one-off interaction inventions
- notebook owns layout
- helper owns logic

## Repair Rules
- fix structural causes, not surface symptoms only
- remove duplicated or conflated responsibilities
- move layout definitions back into the notebook if they leaked into helpers
- move command construction back into helper logic if it leaked into notebook UI code
- normalise helper result shapes if they became inconsistent
- remove speculative features that are outside scope
- preserve working behaviour where compatible with the contract
- prefer regeneration of complete files over incremental patching

## Deliverables
Produce:
1. concise diagnosis of what was wrong
2. repaired complete files
3. short explanation of how the repaired version now matches the contract
4. any intentionally deferred cleanup items

## Acceptance Criteria
The task is complete when:
- the implementation matches the declared architecture
- the notebook UI matches the declared house style if UI exists
- layout is notebook-owned
- helper logic is helper-owned
- distinct functions remain distinct
- scope is reduced back to what was actually requested
- the result is cleaner and easier to reason about

## Output Format
Return:
1. a concise diagnosis
2. assumptions made
3. the complete contents of each new or changed file
4. deferred TODOs or risks
5. any suggested template improvements if the repair exposed prompt ambiguity

Do not return patch fragments unless explicitly requested.

## Notes
Additional context:
[freeform notes]
```

---

## `05-review-template.md

```md
# Codex Review Template

## Template Metadata
- Template name: implementation review
- Template version: v0.3
- Task type: review
- Target area: [module / notebook / control panel / helper]

## Objective
Review the current implementation against the declared contract and identify mismatches, risks, and cleanup priorities.

## Review Scope
Review the implementation for:
- architectural boundary violations
- UI consistency violations
- scope creep
- naming inconsistency
- traceability/logging weakness
- failure-handling weakness
- unnecessary abstraction
- unclear or unstable interaction patterns
- notebook/helper boundary leakage
- result object contract consistency
- command construction boundary violations
- destructive action safety
- refresh behaviour clarity

## Declared Contract
The implementation is expected to preserve:
- [contract point]
- [contract point]
- [contract point]

## UI Review Rules
If user-facing notebook UI exists, review against the UI consistency contract:
- panel order
- button naming
- explicit action invocation
- stable list/table behaviour
- predictable status messages
- refresh behaviour
- avoidance of print-based pseudo-UI
- layout kept in notebook rather than helper modules

## Required Review Output
Return:
1. what matches the contract
2. what violates the contract
3. what is risky but not yet broken
4. what should be fixed first
5. whether the template itself should be improved

## Output Format
Return the review in this structure:

### Matches
- [item]
- [item]

### Violations
- [item]
- [item]

### Risks
- [item]
- [item]

### Recommended next fixes
1. [fix]
2. [fix]
3. [fix]

### Suggested template improvements
- [improvement]
- [improvement]

Do not rewrite code unless explicitly asked.
```

---
## `06-ui-vocabulary.md`

```md
# UI Vocabulary

## Purpose
This file defines the standard language for notebook-based control panels so wording stays consistent across Codex-generated tasks.

The goal is to stop label drift and reduce needless variation.

## Standard Panel Names
Use these names unless there is a clear task-specific reason not to:

- `Header`
- `Actions`
- `Status`
- `Results`
- `Details`

For training/session-control tasks, a deliberate task-specific override is allowed:
- `Launch`
- `Active Sessions`

Do not mix `Launch` and `Actions` as interchangeable generic panel names.

## Standard Button Labels
Prefer:
- `Validate`
- `Launch`
- `Refresh`
- `Stop`
- `Open Log`
- `Run`
- `Save`
- `Apply`
- `Clear`

Avoid:
- `Go`
- `Do It`
- `Execute`
- `Commit`

## Standard Field Labels
Prefer:
- `Session Name`
- `Command Preview`
- `Log Path`
- `Config Path`
- `Status Message`
- `Current Sessions`

Use title case for visible labels unless the notebook already has a different deliberate style.

## Standard Status Categories
Prefer:
- `Info`
- `Validation Error`
- `Success`
- `Failure`
- `Environment Error`

If colour or formatting is used later, it should map cleanly to these categories.

## Standard Table/List Column Names
For session-oriented tools, prefer:
- `Session Name`
- `State`
- `Log Path`
- `Command`
- `Created`
- `Selected`

Only include columns that are actually available and useful.

Do not rename the same concept between iterations without a reason.

## Standard Section Ordering
Unless there is a specific reason to differ:
1. `Header`
2. `Actions`
3. `Status`
4. `Results`

For training/session-control tools, the preferred task-specific ordering is:
1. `Header`
2. `Launch`
3. `Status`
4. `Active Sessions`

## Consistency Rule
If a notebook tool already uses one of these labels, keep reusing it rather than creating a synonym.
```

---

## `07-notebook-widget-style-template.md`

```md
# Notebook Widget Style Template v0.2

## Purpose
This template defines how Codex should structure notebook-based UI work when using widgets.

The goals are:
- keep UI layout easy to tweak by hand
- keep layout decisions in the notebook
- keep helper modules free of presentation structure

## Required Boundary
For notebook UI tasks:

### The notebook file must own:
- widget creation
- layout containers
- grouping into sections/panels
- action-button placement
- message-area placement
- table/list placement
- event wiring from widgets to helper functions

### Helper modules may own:
- typed result objects
- validation functions
- action functions
- query functions
- mapping raw system data into structured Python data

### Helper modules must not own:
- widget container construction
- layout templates
- panel arrangement
- notebook visual composition
- notebook-specific display objects unless explicitly requested

## Preferred Notebook Structure
Use a structure like this in the notebook:

1. imports
2. helper imports
3. widget creation
4. layout assembly
5. render helpers
6. event handlers
7. initial render/refresh

This keeps UI code readable and editable.

## Preferred Widget Sections
Unless explicitly told otherwise, define notebook widgets in these grouped sections:

### Header widgets
- title HTML or markdown display
- subtitle/context text

### Action widgets
- text inputs
- dropdowns/selects
- derived preview displays
- primary action buttons

### Status widgets
- dedicated output/message area
- optionally separate validation and action result areas

### Results widgets
- selectable table/list representation
- refresh button
- optional details view

## Layout Guidance
- Keep section construction explicit in the notebook.
- Prefer readable intermediate variables for sections/panels.
- Do not compress the layout into a single giant expression.
- Keep labels literal and stable.
- Keep action order stable.
- Keep refresh separate from destructive or state-changing actions.

## Render Guidance
Notebook render helpers may:
- take structured helper results
- map them into displayed text/tables/messages
- update widget values/output areas

They should remain notebook-local unless there is a very strong reason not to.

## Event Handler Guidance
Notebook event handlers may:
- gather input values
- call helper functions
- update status/output widgets
- refresh result lists

They should not:
- reimplement helper logic
- parse raw external-system output if helpers can do that instead

## Refresh Guidance
Define refresh explicitly in notebook code:
- what refresh updates
- whether refresh runs automatically after an action
- whether refresh preserves current input values

Default:
- manual refresh exists
- refresh preserves editable input state
- post-action refresh is allowed if it does not wipe user input

## Destructive Action Guidance
For notebook UIs that include destructive actions:
- require explicit selection in the Results area
- require explicit confirmation in notebook UI
- keep destructive buttons visually separate from refresh
- route destructive action logic through helper functions
- report the result in Status

## Codex Instructions for Notebook UI Tasks
When implementing notebook UI:
- put layout in the notebook
- keep helper logic in Python modules
- return complete notebook code, not fragments
- preserve the standard panel order
- use the UI vocabulary file where applicable
- do not hide the layout inside helper abstractions
- do not “improve” maintainability by moving widget layout into helper modules

## Acceptance Signal
A notebook UI implementation follows this template if:
- I can easily tweak the layout directly in the notebook
- the helper module remains UI-agnostic
- the panel structure is obvious in notebook code
- the action wiring is easy to follow
```

---

## `08-training-session-control-template.md`

```md
# Training Session Control Template v0.1

## Template Metadata
- Template name: training session control
- Template version: v0.1
- Task type: feature
- Target area: notebook UI + Python helper

## Purpose
This template is for notebook-based control panels that manage detached training or long-running sessions through a Python helper and a session/persistence layer such as tmux.

It is task-specific and should be preferred over the base template for this tool family.

## Required Pairings
Always use this template together with:
- `02-ui-consistency-contract.md`
- `06-ui-vocabulary.md`
- `07-notebook-widget-style-template.md`
- `09-result-object-contract.md`
- `10-command-construction-contract.md`

## Core Rules
- notebook owns UI layout
- helper owns logic and system interaction
- session name is the primary identifier
- log path is derived deterministically from session name
- duplicate session names are forbidden
- listing sessions, launching sessions, stopping sessions, and validating names are separate functions
- notebook launches only through Python helper calls
- no job resumption
- no checkpoint management
- no log deletion unless explicitly requested

## Architectural Boundaries

### Notebook UI layer
Responsibilities:
- widget creation
- layout for Header, Launch, Status, Active Sessions
- user input for Session Name and other launch controls
- display of Command Preview and Log Path
- explicit buttons for Validate, Launch, Refresh, Stop if included
- event wiring to helper functions
- rendering status messages and session lists

Must not:
- construct tmux commands directly
- parse raw tmux output for business logic
- own command-building rules

### Helper / logic layer
Responsibilities:
- validate session names
- derive log paths from session names
- build launch commands
- list current sessions
- check whether a session already exists
- launch detached sessions
- stop sessions if that feature is explicitly in scope
- return structured results using the standard result object contract

Must not:
- define notebook layout
- return notebook widget structures
- hide command construction inside notebook code

### Session / execution layer
Responsibilities:
- host detached execution
- expose session listing/existence behaviour
- provide external runtime state

Must not:
- act as UI
- absorb helper-layer business rules

## UI Contract
Use this task-specific layout:

1. **Header**
   - title
   - short description

2. **Launch**
   - Session Name input
   - Command Preview display
   - Log Path display
   - Validate button
   - Launch button

3. **Status**
   - validation messages
   - launch success/failure
   - environment/runtime errors

4. **Active Sessions**
   - current sessions list or table
   - Refresh button
   - optional Stop button if stopping sessions is explicitly in scope

Do not substitute a generic Actions panel title here.
Use `Launch` for this tool family.

## Session Name Contract
Use `session_name` as the primary identifier everywhere:
- session identity
- log file stem
- display identity
- helper inputs/outputs

Session name rules:
- non-empty
- trimmed
- safe for session-layer usage
- rejected if obviously invalid
- rejected if already in use

## Log Path Contract
- derive log path deterministically from session name
- do not let notebook layout code invent separate log naming rules
- return derived log path in helper results so the notebook can display it

Example pattern:
- session name: `rb-exp-20260331-01`
- log path: `logs/rb-exp-20260331-01.log`

## Command Construction Contract
Use `10-command-construction-contract.md`.

At minimum, command construction must define:
- what inputs are required
- how session name is injected
- how log path is injected
- how command preview is produced
- where the actual command-building logic lives

## Functional Contract
Required helper functions should remain separate:
- `validate_session_name(...)`
- `list_sessions(...)`
- `session_exists(...)`
- `build_log_path(...)`
- `build_command_preview(...)`
- `build_launch_command(...)`
- `launch_session(...)`
- `stop_session(...)` only if explicitly in scope

Do not collapse these into one “do everything” function.

## Refresh Contract
Define refresh explicitly.

Default rule:
- Refresh updates the Active Sessions view
- Refresh does not wipe current editable launch inputs
- post-launch automatic refresh is allowed
- manual Refresh button must still exist

## Destructive Action Contract
If stopping sessions is in scope:
- require explicit selection in Active Sessions
- require explicit confirmation in notebook UI
- report result in Status
- do not combine Stop with Refresh
- do not place Stop next to Launch without visual separation

## Failure Policy
- reject invalid session names before launch
- reject duplicate session names before launch
- fail clearly if session backend is unavailable
- fail clearly if launch fails
- fail clearly if stop fails
- do not silently continue after partial failure

## Deliverables
Produce:
1. notebook UI code
2. Python helper module or updated helper module
3. any supporting types or utilities
4. short usage note

## Acceptance Criteria
The task is complete when:
- the notebook can display current sessions
- duplicate session names are rejected cleanly
- log path is derived deterministically from session name
- command preview is generated by the helper contract, not improvised in notebook layout code
- a valid new session can be launched through the helper
- notebook layout remains notebook-owned
- helper logic remains helper-owned

## Output Format
Return:
1. a short implementation summary
2. assumptions made
3. the complete contents of each new or materially changed file
4. deferred TODOs or risks
5. any suggested template improvements if the task exposed ambiguity or friction

Do not return partial patches unless explicitly requested.

For very large files that are mostly unchanged:
- return complete contents only for new or materially changed files unless explicitly asked to regenerate the entire file
```

---

## `09-result-object-contract.md`

````md
# Result Object Contract v0.1

## Purpose
This contract defines a small standard result shape for helper-layer functions so notebook event handlers can consume results consistently.

This is not required for every single function, but where a helper returns action status, validation status, or query status, prefer this structure.

## Standard Shape
Use a consistent Python structure equivalent to:

```python
{
    "ok": bool,
    "message": str,
    "data": optional_payload,
    "error_type": optional_error_category,
}
````
```md

A dataclass, TypedDict, or lightweight class is also fine if it preserves the same shape and meaning.

## Field Meanings

- `ok`: whether the operation succeeded
- `message`: short human-readable summary
- `data`: optional structured payload
- `error_type`: optional category such as `validation`, `environment`, `runtime`, `not_found`, `conflict`

## Usage Guidance

Prefer this contract for:

- validation results
- launch results
- stop results
- query results that may fail
- environment checks

Do not force it onto tiny pure helper functions where it adds noise without benefit.

## Notebook Guidance

Notebook event handlers should:

- inspect `ok`
- display `message` in Status
- use `data` to update relevant widgets or tables
- map `error_type` to status styling or grouping if needed

Notebook code should not need to parse free-form text to decide what happened.

## Consistency Rule

If a tool already uses a structured result type, reuse it if it is functionally equivalent to this contract.  
Do not create multiple incompatible result shapes in the same tool.

## Example Categories

Suggested `error_type` values:

- `validation`
- `environment`
- `runtime`
- `conflict`
- `not_found`
- `permission`

````

---

## `10-command-construction-contract.md`

```md
# Command Construction Contract v0.1

## Purpose
This contract defines how process-launch tasks should handle command construction.

The goal is to keep command-building logic explicit, testable, and separate from notebook UI layout.

## Core Rule
Command construction is its own concern.

It must not be:
- improvised inline in notebook layout code
- mixed into widget assembly
- spread across unrelated helper functions without a clear boundary

## Ownership Boundary

### Notebook UI layer may:
- gather user inputs
- request a command preview from the helper
- display the resulting command preview
- trigger launch through helper functions

### Helper / logic layer must:
- validate the inputs required to build the command
- derive any deterministic paths such as log paths
- build the command preview
- build the executable launch command
- inject required identifiers such as session name
- return structured results

## Required Inputs
Define explicitly which inputs command construction uses, for example:
- session name
- config path
- script path
- output/log directory
- optional runtime flags

Do not let notebook layout code invent hidden inputs.

## Session Name Injection
If session identity matters:
- define exactly how `session_name` is used
- keep it consistent between launch command, session creation, and log naming
- do not create separate naming rules in different layers

## Log Path Derivation
If a log path is part of the launch:
- derive it in helper logic
- keep derivation deterministic
- expose it to the notebook for display
- do not compute it independently in both notebook and helper layers

## Command Preview
If the UI displays a command preview:
- the preview should be generated from helper logic or a helper-owned formatting function
- the notebook may render that preview
- the notebook must not invent its own different command-building logic for display only

## Launch Command
The executable launch command may differ slightly from the preview format if needed, but:
- both must be derived from the same underlying inputs and rules
- the difference must be intentional and understandable

## Traceability
Where relevant, command construction should make it possible to answer:
- what command was intended
- what session name was used
- what log path was used
- which inputs produced that command

## Anti-Patterns
Avoid:
- building commands inline inside button callbacks
- building preview text separately from actual launch commands with different rules
- deriving log path independently in multiple layers
- mixing session validation and command construction into one opaque function

## Minimal Function Split
For launch tools, prefer a split like:
- input validation
- log path derivation
- command preview construction
- executable command construction
- launch action

Do not collapse all of this into one helper unless there is a strong reason.
````

---
