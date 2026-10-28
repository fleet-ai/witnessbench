# Security policy

Do not report a security problem in a public issue. This includes a leak of the private test games and any way to
read them through this code. Report it privately instead: open the repository's
[security page](https://github.com/fleet-ai/witnessbench/security) and choose **Report a vulnerability**. Include:

- what is affected (a file, a command or a setting) and the commit you used;
- the steps or the code that reproduce the problem;
- what an attacker could do with it.

`witness_game` supplies no authentication and starts no server: a game service built on it must run behind a
transport that authenticates its callers ([harness guide](docs/harness.md#playing-a-game-in-another-process)).
