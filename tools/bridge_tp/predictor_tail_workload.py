"""Preregister concrete implementation workloads for a six-request tail pilot."""

VERSION = "tail6-workload-v2-20261011"

# Each unit specifies distinct behavior; workloads do not prescribe output length.
TASKS = [
    {
        "lang": "en",
        "target": "12289-16384",
        "family": "compiler_interpreter",
        "title": "A typed scripting language with a bytecode virtual machine",
        "modules": [
            "source.py: source spans, line/column indexing, diagnostic rendering",
            "tokens.py: token kinds, token records, literal decoding",
            "lexer.py: comments, escaped strings, numeric tokens, invalid characters",
            "ast_nodes.py: expressions, statements, functions, blocks, visitor API",
            "parser.py: precedence, calls, indexing, if/while/return, syntax errors",
            "symbols.py: nested scopes, bindings, duplicate and unknown names",
            "typesys.py: int/bool/string/list types, checking, function signatures",
            "bytecode.py: opcodes, constants, instruction records, source mapping",
            "compiler.py: expression lowering, branch patching, loops, function calls",
            "verifier.py: stack effects, branch targets, invalid bytecode diagnostics",
            "vm.py: call frames, stack values, arithmetic, collections, runtime errors",
            "builtins.py: print, len, range, conversion and argument validation",
            "optimizer.py: constant folding and dead branches with effect preservation",
            "disassemble.py: readable listings and source-position annotations",
            "cli.py: compile/run/disassemble commands and structured error reporting",
            "tests.py: isolated module tests and complete source-to-result tests",
        ],
        "cases": [
            "precedence and associativity; nested scopes; factorial recursion",
            "Fibonacci loop; list indexing and bounds errors; escaped strings",
            "wrong argument count; illegal return; duplicate name; unknown variable",
            "type mismatch; division by zero; invalid opcode; bad jump target",
            "stack underflow; nested function calls; optimized/unoptimized agreement",
            "line/column errors; comment handling; CLI compile/run round trip",
        ],
        "spec": (
            "Use let name: type = expression; function declarations with typed "
            "parameters and return types; braces for blocks. Define a complete "
            "grammar first. Keep compilation and execution separate. Use Python "
            "3.12 standard library only and provide executable unittest fixtures."
        ),
    },
    {
        "lang": "zh",
        "target": "12289-16384",
        "family": "compiler_interpreter",
        "title": "带查询规划与可验证执行器的关系查询语言",
        "modules": [
            "errors.py：词法、语法、绑定、执行错误及位置显示",
            "tokens.py：关键字、标识符、字符串和数值的令牌协议",
            "lexer.py：注释、转义、比较运算符与错误定位",
            "ast_nodes.py：SELECT、表达式、JOIN、聚合及排序节点",
            "parser.py：完整优先级、FROM/JOIN/WHERE/GROUP/ORDER/LIMIT语法",
            "catalog.py：表模式、别名解析、字段歧义、数据加载",
            "binder.py：名字绑定、类型检查、聚合合法性检查",
            "logical.py：扫描、过滤、投影、连接、分组及排序计划",
            "rewrite.py：常量折叠、过滤下推、投影裁剪及正确性条件",
            "physical.py：执行节点接口、计划打印和运行统计",
            "scan_filter.py：扫描、谓词求值、NULL三值逻辑",
            "joins.py：嵌套循环与哈希连接、重复键和空表",
            "aggregate.py：COUNT/SUM/AVG、空输入、分组键和NULL处理",
            "sort_limit.py：多字段排序、NULL顺序、稳定性和LIMIT",
            "cli.py：查询命令、CSV导入、输出格式与失败状态",
            "tests.py：单元测试、查询结果验证和优化前后等价验证",
        ],
        "cases": [
            "基础投影；复杂布尔条件；字段别名；同名字段歧义",
            "两个表连接；重复连接键；空表连接；NULL比较",
            "COUNT(*)与COUNT(column)；分组SUM；无输入聚合；多字段排序",
            "LIMIT边界；未知表；未知列；错误语法；非法聚合",
            "CSV类型转换；过滤下推前后等价；连接算法结果等价；CLI端到端",
        ],
        "spec": (
            "先定义支持的SQL子集和NULL语义，不使用sqlite或现成SQL解析器。"
            "使用Python3.12标准库。测试采用明确的customers、orders、items"
            "小表，逐个给出输入查询、预期行和可执行断言。"
        ),
    },
    {
        "lang": "en",
        "target": "8193-12288",
        "family": "compiler_interpreter",
        "title": "A sandboxed template interpreter with dependency tracking",
        "modules": [
            "source.py: source positions and diagnostics",
            "lexer.py: literal text, interpolation, block tags, escaping",
            "expressions.py: arithmetic, comparisons, paths, safe function calls",
            "nodes.py: text, expression, if/else, for, include, macro nodes",
            "parser.py: nested blocks, tag matching, malformed-input handling",
            "context.py: scoped variables, missing values, loop metadata",
            "evaluator.py: whitelisted operations without Python eval or exec",
            "renderer.py: node execution, HTML escaping, nested macro calls",
            "loader.py: directory loading, relative includes, traversal rejection",
            "dependencies.py: include graph, cycle reports, invalidation propagation",
            "cache.py: file fingerprints, compiled-template cache, change detection",
            "cli.py: render/check/dependencies commands with useful errors",
            "tests.py: expression, parser, renderer and filesystem fixtures",
            "demo.py: integrated example rendering several linked templates",
        ],
        "cases": [
            "plain text; interpolation; escaping; nested if/else; empty loops",
            "loop metadata; nested include; macro parameters; missing variable",
            "bad expression; unclosed tag; include cycle; forbidden path",
            "cache hit; cache invalidation; safe-call rejection; CLI fixture",
        ],
        "spec": (
            "Use {{ expression }} and {% directive %} syntax. Specify the grammar "
            "and error behavior. Use Python 3.12 standard library only. The "
            "filesystem tests must use temporary directories and complete fixtures."
        ),
    },
    {
        "lang": "zh",
        "target": "8193-12288",
        "family": "data_system",
        "title": "具备事件时间窗口与恢复能力的流式日志处理引擎",
        "modules": [
            "schema.py：事件协议、字段类型、版本和序列化",
            "parse.py：文本与JSON日志解析、时间区和错误记录",
            "validate.py：必需字段、范围检查、失败隔离",
            "expressions.py：过滤表达式的令牌、语法树和安全求值",
            "sources.py：文件输入、偏移记录、断点恢复",
            "watermarks.py：事件时间、水位推进、乱序与迟到规则",
            "windows.py：滚动窗口、滑动窗口、窗口归属和关闭",
            "aggregate.py：计数、总和、平均值、分组状态",
            "deduplicate.py：事件标识去重、过期和恢复",
            "state.py：磁盘状态、批次提交、损坏检查",
            "checkpoint.py：快照协议、原子替换、恢复校验",
            "sinks.py：JSON与CSV输出、重复提交防护",
            "pipeline.py：完整事件处理、指标和错误传播",
            "cli.py：配置、运行、恢复、统计命令",
            "tests.py：乱序、恢复、重复事件与完整流水线测试",
        ],
        "cases": [
            "正常日志；字段缺失；非法时间；时区转换；复杂过滤",
            "窗口边界；乱序事件；迟到事件；空窗口；多组聚合",
            "重复标识；去重过期；输入偏移恢复；快照损坏",
            "写出失败；重复提交；中途崩溃恢复；滑动窗口重叠",
            "配置错误；完整输入文件到预期聚合输出",
        ],
        "spec": (
            "使用Python3.12标准库。先定义水位、迟到、重复以及恢复语义。"
            "每个窗口案例给出确定的时间戳事件序列和预期聚合结果。"
            "用临时目录模拟失败与恢复，不连接外部服务。"
        ),
    },
    {
        "lang": "en",
        "target": "12289-16384",
        "family": "data_system",
        "title": "A versioned local column store with transactional snapshots",
        "modules": [
            "schema.py: typed columns, defaults, nullability, schema versions",
            "encoding.py: null bitmap, integer/string encoding, decoding checks",
            "segments.py: immutable segment layout, headers, checksums, iteration",
            "manifest.py: snapshot manifests and segment references",
            "ingest.py: CSV/JSON records, validation, batching, rejected rows",
            "statistics.py: column statistics and safe pruning predicates",
            "expressions.py: typed filter expressions and vector evaluation",
            "scan.py: snapshot reads, pruning, row masks, projection",
            "operators.py: hash aggregation, joins, external sorting",
            "transaction.py: append/delete staging and atomic commit",
            "locking.py: writer coordination, stale locks, conflict detection",
            "recovery.py: incomplete commits, temporary files, damaged manifests",
            "compact.py: segment merging, deleted rows and old snapshot retention",
            "migration.py: adding columns, defaults, compatible schema changes",
            "cli.py: create/import/query/snapshot/compact commands",
            "tests.py: storage round trips, failure injection and snapshot fixtures",
        ],
        "cases": [
            "integer round trip; strings and Unicode; NULL values; empty segment",
            "bad checksum; truncated file; rejected CSV row; projection/filter",
            "statistics pruning; group aggregation; duplicate join keys; sorting",
            "snapshot isolation; concurrent-writer conflict; interrupted commit",
            "recovery after manifest publication; compaction with live snapshot",
            "schema default; invalid migration; CLI import/query round trip",
        ],
        "spec": (
            "Define binary and JSON formats precisely. Use Python 3.12 standard "
            "library only, without sqlite, pandas or pyarrow. Include small exact "
            "records for tests and explain the crash-consistency invariants."
        ),
    },
    {
        "lang": "zh",
        "target": "8193-12288",
        "family": "data_system",
        "title": "可复现计算流水线与实验结果追踪系统",
        "modules": [
            "spec.py：任务协议、输入输出、参数与配置校验",
            "graph.py：依赖图、拓扑排序、环与缺失依赖诊断",
            "parameters.py：参数网格展开、确定性任务标识",
            "artifacts.py：内容哈希、文件存储、元数据和校验",
            "registry.py：任务与结果持久化、状态迁移和查询",
            "cache.py：缓存键、命中条件、输入变化的失效传播",
            "scheduler.py：就绪任务、失败依赖、资源额度和执行顺序",
            "worker.py：本地执行、超时、异常和结果回传",
            "retry.py：重试上限、幂等输出和永久失败",
            "checkpoint.py：运行快照、恢复和已完成任务去重",
            "metrics.py：指标协议、汇总、异常值和缺失值",
            "report.py：结果表、运行对照、JSON与CSV导出",
            "cli.py：创建、运行、恢复、查看、对照命令",
            "tests.py：图算法、缓存、错误恢复与完整小实验",
        ],
        "cases": [
            "简单依赖链；分叉合流；依赖环；缺失任务；参数网格",
            "相同输入缓存命中；输入变化；配置变化；损坏产物",
            "任务异常；超时；重试成功；永久失败；依赖失败",
            "中途恢复；结果去重；资源额度；缺失指标",
            "确定性排序；CLI执行完整示例并核对预期结果表",
        ],
        "spec": (
            "使用Python3.12标准库。示例任务用确定性的小规模算术和文件转换，"
            "不调用外部服务。明确每条失败路径对应的状态变化，测试用临时目录"
            "和可注入的时钟，不依赖真实长时间等待。"
        ),
    },
]


def recipe():
    """Return six independent tasks with enumerated modules and acceptance cases."""
    rows = []
    for number, task in enumerate(TASKS, 1):
        modules = "\n".join(
            f"{index}. {unit}" for index, unit in enumerate(task["modules"], 1)
        )
        cases = "\n".join(f"- {case}" for case in task["cases"])
        if task["lang"] == "en":
            prompt = (
                f"Implement this complete project: {task['title']}.\n"
                f"Specification: {task['spec']}\n"
                f"Required modules and behaviors:\n{modules}\n"
                f"Required acceptance cases:\n{cases}\n"
                "Provide the grammar or data format and public interfaces first. "
                "Then emit a separate, fully implemented code block for EVERY "
                "listed module, with its filename. Implement each listed behavior "
                "and its error handling explicitly; do not replace implementations "
                "with TODO, pass, ellipses, pseudocode, or 'same as above'. Emit "
                "each module once, using consistent cross-module imports. After "
                "the modules, supply executable unittest cases for every listed "
                "acceptance case, including concrete inputs, expected results and "
                "assertions. Explain each module's invariants and a worked example, "
                "then give the command and expected output for an integrated demo. "
                "This is one complete response, not an outline or a multi-turn "
                "continuation. Finish naturally when all deliverables are complete."
            )
        else:
            prompt = (
                f"请完整实现项目：{task['title']}。\n"
                f"具体约束：{task['spec']}\n"
                f"必须逐个实现的模块及行为：\n{modules}\n"
                f"必须逐个覆盖的验收案例：\n{cases}\n"
                "先说明语法或数据格式以及公共接口。随后给每一个列出的模块输出"
                "独立的完整代码块，并标注文件名；明确实现各项行为和错误处理，"
                "不能用TODO、pass、省略号、伪代码或‘同上’替代实现。每个模块"
                "只输出一次，跨模块导入与接口必须一致。模块之后给出可执行的"
                "unittest测试，逐个覆盖列出的验收案例，包含具体输入、预期结果"
                "和断言。解释各模块的不变量和一个完整案例，最后提供综合示例的"
                "运行命令和预期结果。本次一次性输出实际交付物，不是目录或"
                "多轮续写计划。完成全部交付物后自然结束。"
            )
        identity = f"predictor-{VERSION}-task{number:02d}"
        rows.append(
            {
                "id": identity,
                "source_tree_id": identity,
                "source": "constructed_long_output_workload_recipe_pilot",
                "split": "pilot",
                "collection_role": "recipe_development_not_formal_holdout",
                "lang": task["lang"],
                "task_kind": task["family"],
                "target_output_range": task["target"],
                "title": task["title"],
                "required_modules": task["modules"],
                "acceptance_case_groups": task["cases"],
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a careful software engineer.",
                    },
                    {"role": "user", "content": prompt},
                ],
            }
        )
    return rows
