"""
主程序入口 - 晶圆异常数据检索系统

功能:

1. 精确搜索
   Payload字段过滤

2. 混合智能搜索
   Qwen3向量
   +
   BM25关键词
   +
   字符TF-IDF
   +
   Fusion
   +
   Rerank

3. 智能路由
   自动判断精确查询/混合查询

4. 模块化设计
"""


from config import (
    DEFAULT_TOP_K,
    SCORE_THRESHOLD
)


from utils.qdrant_client import (
    get_qdrant_manager
)


from utils.result_formatter import (
    get_result_formatter
)


from searchers.hybrid_searcher import (
    get_hybrid_searcher
)





def check_services():

    """
    检查服务连接状态
    """

    qdrant = (
        get_qdrant_manager()
    )


    result = (
        qdrant.check_connection()
    )



    print("=" * 60)

    print(
        "晶圆异常数据检索系统"
    )

    print("=" * 60)



    if result["success"]:


        print(
            f"✓ {result['message']}"
        )


        print(
            f"  Qdrant向量数量: "
            f"{result['points_count']}"
        )


        return True



    else:


        print(
            f"✗ {result['message']}"
        )


        return False





def show_help():


    print("\n" + "=" * 60)

    print(
        "查询方式说明"
    )

    print("=" * 60)



    print(
        "\n【1. 精确查询】"
    )


    print(
        "输入包含明确字段的信息，"
        "系统自动字段匹配:"
    )


    print(
        "例如:"
    )


    print(
        "DN编号: DN-20241020-01"
    )


    print(
        "LotID: BB84606"
    )


    print(
        "平台: CL040LP"
    )


    print(
        "缺陷类型: Residue"
    )


    print(
        "组合:"
        "平台CL040LP 缺陷类型Residue"
    )



    print(
        "\n【2. 混合智能查询】"
    )


    print(
        "系统自动融合:"
    )


    print(
        "Qwen3向量"
        "+"
        "BM25关键词"
        "+"
        "字符TF-IDF"
        "+"
        "Fusion"
        "+"
        "Rerank"
    )



    print(
        "例如:"
    )


    print(
        "查找Bottom Right位置Particle异常"
    )


    print(
        "CL040LP平台最近的异常"
    )


    print(
        "类似Residue缺陷的问题"
    )



    print(
        "\n【3. 特殊命令】"
    )


    print(
        "explain: 查询内容"
        " -> 查看查询解析"
    )


    print(
        "help -> 显示帮助"
    )


    print(
        "exit/quit/q -> 退出"
    )






def main():


    if not check_services():

        return



    try:


        searcher = (
            get_hybrid_searcher()
        )


        formatter = (
            get_result_formatter()
        )



    except ImportError as e:


        print(
            f"模块加载失败:{e}"
        )


        return




    print(
        "\n输入 help 查看查询方式"
    )


    print(
        "输入 exit 退出程序"
    )




    while True:


        try:


            query = input(

                "\n请输入查询内容: "

            ).strip()



        except (
            EOFError,
            KeyboardInterrupt
        ):


            print(
                "\n程序结束"
            )


            break




        if not query:

            continue



        cmd=query.lower()



        if cmd in (
            "exit",
            "quit",
            "q"
        ):


            print(
                "程序已退出"
            )


            break




        if cmd=="help":


            show_help()

            continue





        # ==========================
        # explain模式
        # ==========================


        if query.lower().startswith(
            "explain:"
        ):


            explain_query=(

                query[8:]

                .strip()

            )



            print(
                "\n"
                +
                "="*60
            )


            print(
                "查询解析过程"
            )


            print(
                "="*60
            )



            try:


                explanation=(

                    searcher
                    .explain_query(
                        explain_query
                    )

                )



                print(

                    f"\n查询内容:"
                    f"{explanation['query']}"

                )


                print(

                    f"\n提取字段:"
                    f"{explanation['conditions']}"

                )


                print(

                    f"\n搜索策略:"
                    f"{explanation['strategy']}"

                )



            except Exception as e:


                print(
                    f"解析失败:{e}"
                )



            continue






        # ==========================
        # 正常搜索
        # ==========================


        try:


            results, elapsed_ms, search_mode=(

                searcher.search(

                    query=query,

                    top_k=
                    DEFAULT_TOP_K,

                    score_threshold=
                    SCORE_THRESHOLD

                )

            )



            formatter.format_results(

                results,

                elapsed_ms,

                search_mode

            )




        except ValueError as e:


            print(
                f"输入错误:{e}"
            )



        except RuntimeError as e:


            print(
                f"查询失败:{e}"
            )



        except Exception as e:


            print(
                f"未识别异常:{e}"
            )







if __name__=="__main__":


    main()