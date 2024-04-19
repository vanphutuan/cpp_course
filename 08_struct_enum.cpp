#include<iostream>
#include<string>
using namespace std;

enum education {th, thcs, thpt, dh, ch, ncs};
enum gender {female, male, other};
struct person
{
    string ten;
    int tuoi;
    education giaoduc;
    gender gioitinh;
};
int main(){
    int gt, td;
    person person01; 
    cout << "Nhap ten: \n\t";
    getline(cin,person01.ten);
    cout << "Nhap tuoi: \n\t";
    cin >> person01.tuoi;
    cout << "Nhap gioi tinh {0: female; 1: male; -: other}: \n\t";
    cin >> gt;
    switch (gt)
    {
    case 0:
        person01.gioitinh = female;
        break;
    case 1:
        person01.gioitinh = male;
        break;
    default:
        person01.gioitinh = other;
        break;
    }
    cout << "Nhap trinh do giao duc" << endl;
    cout << "\t1-tieu hoc\n\t2-trung hoc co so"<<
            "\n\t3-trung hoc pho thong"<<
            "\n\t4-dai hoc\n\t5-cao hoc" <<
            "\n\t6-nghien cuu sinh" << endl;
    cout << "so ban chon: \n\t";
    cin >> td;
}
