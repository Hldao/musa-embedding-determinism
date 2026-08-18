# Third-party notices

This repository contains an independently prepared reproducer, benchmark,
analysis, regression test proposal, and a small source patch for
[`MooreThreads/torch_musa`](https://github.com/MooreThreads/torch_musa).

## Scope of WKK's contribution

WKK's original contribution is the problem isolation, deterministic-routing
change, test case, benchmark harness, documentation, and collected synthetic
test results. The patch necessarily reproduces limited surrounding lines from
the upstream `torch_musa` source so that the change can be reviewed and
applied. WKK does not claim ownership of those upstream lines.

The patch was developed and validated against the public `torch_musa` v1.3.0
source lineage (tag commit `73c9f5ba21dc6a0dc0ed07ef027c804dc53644a9`).
It is a proposed community contribution and is not an official Moore Threads
release. The names of upstream projects and copyright holders are used only for
identification and must not be taken as endorsement.

WKK's original files and additions are offered under the repository's
BSD-3-Clause license. Any copied or modified upstream material continues to be
governed by its existing notices and license conditions, reproduced below.

## torch_musa upstream license

Source: <https://github.com/MooreThreads/torch_musa/blob/v1.3.0/LICENSE>

The following text is reproduced from the upstream v1.3.0 `LICENSE` file:

```text
BSD 3-Clause License

Copyright (c) 2023 , Moore Threads Technology  Co., Ltd. 
Copyright (c) 2022, Facebook Inc. and the respective contributors 
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

------------------------------------------------------------------------------------
This product bundles various third-party components under other open source licenses.
This section summarizes those components and their licenses. See licenses/
for text of these licenses.


License in PyToch(https://github.com/pytorch/pytorch/blob/main/LICENSE)
-----------------
tools/setup_helper
torch_musa/csrc/
torch_musa/core


Apache Software Foundation License 2.0
--------------------------------------
tools/lint


BSD 2-clause License
--------------------
docs


Apache Software Foundation License 2.0
--------------------------------------
examples/cpp
```

No MUSA SDK binaries, vendor wheels, model weights, training data, or private
training artifacts are included or licensed by this repository. Users must
obtain required runtimes separately and comply with their applicable terms.
