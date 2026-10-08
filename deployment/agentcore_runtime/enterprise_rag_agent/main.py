
def get_status():
    return {
        "application": "Enterprise RAG Agent",
        "status": "running",
        "version": "0.1.0"
    }


def main():
    status = get_status()
    print(status)


if __name__ == "__main__":
    main()
